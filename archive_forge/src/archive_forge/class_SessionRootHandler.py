import asyncio
import json
from jupyter_client.kernelspec import NoSuchKernel
from jupyter_core.utils import ensure_async
from tornado import web
from jupyter_server.auth.decorator import authorized
from jupyter_server.utils import url_path_join
from ...base.handlers import APIHandler
class SessionRootHandler(SessionsAPIHandler):
    """A Session Root API handler."""

    @web.authenticated
    @authorized
    async def get(self):
        """Get a list of running sessions."""
        sm = self.session_manager
        sessions = await ensure_async(sm.list_sessions())
        self.finish(json.dumps(sessions, default=json_default))

    @web.authenticated
    @authorized
    async def post(self):
        """Create a new session."""
        sm = self.session_manager
        model = self.get_json_body()
        if model is None:
            raise web.HTTPError(400, 'No JSON data provided')
        if 'notebook' in model:
            self.log.warning('Sessions API changed, see updated swagger docs')
            model['type'] = 'notebook'
            if 'name' in model['notebook']:
                model['path'] = model['notebook']['name']
            elif 'path' in model['notebook']:
                model['path'] = model['notebook']['path']
        try:
            path = model['path']
        except KeyError as e:
            raise web.HTTPError(400, 'Missing field in JSON data: path') from e
        try:
            mtype = model['type']
        except KeyError as e:
            raise web.HTTPError(400, 'Missing field in JSON data: type') from e
        name = model.get('name', None)
        kernel = model.get('kernel', {})
        kernel_name = kernel.get('name', None)
        kernel_id = kernel.get('id', None)
        if not kernel_id and (not kernel_name):
            self.log.debug('No kernel specified, using default kernel')
            kernel_name = None
        exists = await ensure_async(sm.session_exists(path=path))
        if exists:
            s_model = await sm.get_session(path=path)
        else:
            try:
                s_model = await sm.create_session(path=path, kernel_name=kernel_name, kernel_id=kernel_id, name=name, type=mtype)
            except NoSuchKernel:
                msg = "The '%s' kernel is not available. Please pick another suitable kernel instead, or install that kernel." % kernel_name
                status_msg = '%s not found' % kernel_name
                self.log.warning('Kernel not found: %s' % kernel_name)
                self.set_status(501)
                self.finish(json.dumps({'message': msg, 'short_message': status_msg}))
                return
            except Exception as e:
                raise web.HTTPError(500, str(e)) from e
        location = url_path_join(self.base_url, 'api', 'sessions', s_model['id'])
        self.set_header('Location', location)
        self.set_status(201)
        self.finish(json.dumps(s_model, default=json_default))