import io
import os
import sys
import zipfile
from anyio.to_thread import run_sync
from jupyter_core.utils import ensure_async
from nbformat import from_dict
from tornado import web
from tornado.log import app_log
from jupyter_server.auth.decorator import authorized
from ..base.handlers import FilesRedirectHandler, JupyterHandler, path_regex
class NbconvertFileHandler(JupyterHandler):
    """An nbconvert file handler."""
    auth_resource = AUTH_RESOURCE
    SUPPORTED_METHODS = ('GET',)

    @web.authenticated
    @authorized
    async def get(self, format, path):
        """Get a notebook file in a desired format."""
        self.check_xsrf_cookie()
        exporter = get_exporter(format, config=self.config, log=self.log)
        path = path.strip('/')
        if hasattr(self.contents_manager, '_get_os_path'):
            os_path = self.contents_manager._get_os_path(path)
            ext_resources_dir, basename = os.path.split(os_path)
        else:
            ext_resources_dir = None
        model = await ensure_async(self.contents_manager.get(path=path))
        name = model['name']
        if model['type'] != 'notebook':
            return FilesRedirectHandler.redirect_to_files(self, path)
        nb = model['content']
        self.set_header('Last-Modified', model['last_modified'])
        mod_date = model['last_modified'].strftime(date_format)
        nb_title = os.path.splitext(name)[0]
        resource_dict = {'metadata': {'name': nb_title, 'modified_date': mod_date}, 'config_dir': self.application.settings['config_dir']}
        if ext_resources_dir:
            resource_dict['metadata']['path'] = ext_resources_dir
        try:
            output, resources = await run_sync(lambda: exporter.from_notebook_node(nb, resources=resource_dict))
        except Exception as e:
            self.log.exception('nbconvert failed: %r', e)
            raise web.HTTPError(500, 'nbconvert failed: %s' % e) from e
        if respond_zip(self, name, output, resources):
            return None
        if self.get_argument('download', 'false').lower() == 'true':
            filename = os.path.splitext(name)[0] + resources['output_extension']
            self.set_attachment_header(filename)
        if exporter.output_mimetype:
            self.set_header('Content-Type', '%s; charset=utf-8' % exporter.output_mimetype)
        self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.finish(output)