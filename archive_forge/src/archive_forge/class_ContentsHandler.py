import json
from http import HTTPStatus
from typing import Any, Dict, List
from jupyter_core.utils import ensure_async
from tornado import web
from jupyter_server.auth.decorator import allow_unauthenticated, authorized
from jupyter_server.base.handlers import APIHandler, JupyterHandler, path_regex
from jupyter_server.utils import url_escape, url_path_join
class ContentsHandler(ContentsAPIHandler):
    """A contents handler."""

    def location_url(self, path):
        """Return the full URL location of a file.

        Parameters
        ----------
        path : unicode
            The API path of the file, such as "foo/bar.txt".
        """
        return url_path_join(self.base_url, 'api', 'contents', url_escape(path))

    def _finish_model(self, model, location=True):
        """Finish a JSON request with a model, setting relevant headers, etc."""
        if location:
            location = self.location_url(model['path'])
            self.set_header('Location', location)
        self.set_header('Last-Modified', model['last_modified'])
        self.set_header('Content-Type', 'application/json')
        self.finish(json.dumps(model, default=json_default))

    async def _finish_error(self, code, message):
        """Finish a JSON request with an error code and descriptive message"""
        self.set_status(code)
        self.write(message)
        await self.finish()

    @web.authenticated
    @authorized
    async def get(self, path=''):
        """Return a model for a file or directory.

        A directory model contains a list of models (without content)
        of the files and directories it contains.
        """
        path = path or ''
        cm = self.contents_manager
        type = self.get_query_argument('type', default=None)
        if type not in {None, 'directory', 'file', 'notebook'}:
            type = 'file'
        format = self.get_query_argument('format', default=None)
        if format not in {None, 'text', 'base64'}:
            raise web.HTTPError(400, 'Format %r is invalid' % format)
        content_str = self.get_query_argument('content', default='1')
        if content_str not in {'0', '1'}:
            raise web.HTTPError(400, 'Content %r is invalid' % content_str)
        content = int(content_str or '')
        hash_str = self.get_query_argument('hash', default='0')
        if hash_str not in {'0', '1'}:
            raise web.HTTPError(400, f'Content {hash_str!r} is invalid')
        require_hash = int(hash_str)
        if not cm.allow_hidden and await ensure_async(cm.is_hidden(path)):
            await self._finish_error(HTTPStatus.NOT_FOUND, f'file or directory {path!r} does not exist')
        try:
            expect_hash = require_hash
            try:
                model = await ensure_async(self.contents_manager.get(path=path, type=type, format=format, content=content, require_hash=require_hash))
            except TypeError:
                expect_hash = False
                model = await ensure_async(self.contents_manager.get(path=path, type=type, format=format, content=content))
            validate_model(model, expect_content=content, expect_hash=expect_hash)
            self._finish_model(model, location=False)
        except web.HTTPError as exc:
            if exc.status_code == HTTPStatus.NOT_FOUND:
                await self._finish_error(HTTPStatus.NOT_FOUND, f'file or directory {path!r} does not exist')
            raise

    @web.authenticated
    @authorized
    async def patch(self, path=''):
        """PATCH renames a file or directory without re-uploading content."""
        cm = self.contents_manager
        model = self.get_json_body()
        if model is None:
            raise web.HTTPError(400, 'JSON body missing')
        old_path = model.get('path')
        if old_path and (not cm.allow_hidden) and (await ensure_async(cm.is_hidden(path)) or await ensure_async(cm.is_hidden(old_path))):
            raise web.HTTPError(400, f'Cannot rename file or directory {path!r}')
        model = await ensure_async(cm.update(model, path))
        validate_model(model)
        self._finish_model(model)

    async def _copy(self, copy_from, copy_to=None):
        """Copy a file, optionally specifying a target directory."""
        self.log.info('Copying {copy_from} to {copy_to}'.format(copy_from=copy_from, copy_to=copy_to or ''))
        model = await ensure_async(self.contents_manager.copy(copy_from, copy_to))
        self.set_status(201)
        validate_model(model)
        self._finish_model(model)

    async def _upload(self, model, path):
        """Handle upload of a new file to path"""
        self.log.info('Uploading file to %s', path)
        model = await ensure_async(self.contents_manager.new(model, path))
        self.set_status(201)
        validate_model(model)
        self._finish_model(model)

    async def _new_untitled(self, path, type='', ext=''):
        """Create a new, empty untitled entity"""
        self.log.info('Creating new %s in %s', type or 'file', path)
        model = await ensure_async(self.contents_manager.new_untitled(path=path, type=type, ext=ext))
        self.set_status(201)
        validate_model(model)
        self._finish_model(model)

    async def _save(self, model, path):
        """Save an existing file."""
        chunk = model.get('chunk', None)
        if not chunk or chunk == -1:
            self.log.info('Saving file at %s', path)
        model = await ensure_async(self.contents_manager.save(model, path))
        validate_model(model)
        self._finish_model(model)

    @web.authenticated
    @authorized
    async def post(self, path=''):
        """Create a new file in the specified path.

        POST creates new files. The server always decides on the name.

        POST /api/contents/path
          New untitled, empty file or directory.
        POST /api/contents/path
          with body {"copy_from" : "/path/to/OtherNotebook.ipynb"}
          New copy of OtherNotebook in path
        """
        cm = self.contents_manager
        file_exists = await ensure_async(cm.file_exists(path))
        if file_exists:
            raise web.HTTPError(400, 'Cannot POST to files, use PUT instead.')
        model = self.get_json_body()
        if model:
            copy_from = model.get('copy_from')
            if copy_from:
                if not cm.allow_hidden and (await ensure_async(cm.is_hidden(path)) or await ensure_async(cm.is_hidden(copy_from))):
                    raise web.HTTPError(400, f'Cannot copy file or directory {path!r}')
                else:
                    await self._copy(copy_from, path)
            else:
                ext = model.get('ext', '')
                type = model.get('type', '')
                if type not in {None, '', 'directory', 'file', 'notebook'}:
                    type = 'file'
                await self._new_untitled(path, type=type, ext=ext)
        else:
            await self._new_untitled(path)

    @web.authenticated
    @authorized
    async def put(self, path=''):
        """Saves the file in the location specified by name and path.

        PUT is very similar to POST, but the requester specifies the name,
        whereas with POST, the server picks the name.

        PUT /api/contents/path/Name.ipynb
          Save notebook at ``path/Name.ipynb``. Notebook structure is specified
          in `content` key of JSON request body. If content is not specified,
          create a new empty notebook.
        """
        model = self.get_json_body()
        cm = self.contents_manager
        if model:
            if model.get('copy_from'):
                raise web.HTTPError(400, 'Cannot copy with PUT, only POST')
            if not cm.allow_hidden and (model.get('path') and await ensure_async(cm.is_hidden(model.get('path'))) or await ensure_async(cm.is_hidden(path))):
                raise web.HTTPError(400, f'Cannot create file or directory {path!r}')
            exists = await ensure_async(self.contents_manager.file_exists(path))
            if model.get('type', '') not in {None, '', 'directory', 'file', 'notebook'}:
                model['type'] = 'file'
            if exists:
                await self._save(model, path)
            else:
                await self._upload(model, path)
        else:
            await self._new_untitled(path)

    @web.authenticated
    @authorized
    async def delete(self, path=''):
        """delete a file in the given path"""
        cm = self.contents_manager
        if not cm.allow_hidden and await ensure_async(cm.is_hidden(path)):
            raise web.HTTPError(400, f'Cannot delete file or directory {path!r}')
        self.log.warning('delete %s', path)
        await ensure_async(cm.delete(path))
        self.set_status(204)
        self.finish()