from urllib.parse import urlencode
from paste.httpexceptions import HTTPFound
from paste.httpheaders import CONTENT_LENGTH
from paste.httpheaders import CONTENT_TYPE
from paste.httpheaders import LOCATION
from paste.request import construct_url
from paste.request import parse_dict_querystring
from paste.request import parse_formvars
from repoze.who.interfaces import IChallenger
from repoze.who.interfaces import IIdentifier
from repoze.who.plugins.form import FormPlugin
from zope.interface import implements
class FormHiddenPlugin(FormPlugin):
    implements(IChallenger, IIdentifier)

    def identify(self, environ):
        logger = environ.get('repoze.who.logger', '')
        logger.info('formplugin identify')
        query = parse_dict_querystring(environ)
        if query.get(self.login_form_qs):
            form = parse_formvars(environ)
            from StringIO import StringIO
            environ['wsgi.input'] = StringIO()
            form.update(query)
            qinfo = {}
            for key, val in form.items():
                if key.startswith('_') and key.endswith('_'):
                    qinfo[key[1:-1]] = val
            if qinfo:
                environ['s2repoze.qinfo'] = qinfo
            try:
                login = form['login']
                password = form['password']
            except KeyError:
                return None
            del query[self.login_form_qs]
            query.update(qinfo)
            environ['QUERY_STRING'] = urlencode(query)
            environ['repoze.who.application'] = HTTPFound(construct_url(environ))
            credentials = {'login': login, 'password': password}
            max_age = form.get('max_age', None)
            if max_age is not None:
                credentials['max_age'] = max_age
            return credentials
        return None

    def challenge(self, environ, status, app_headers, forget_headers):
        logger = environ.get('repoze.who.logger', '')
        logger.info('formplugin challenge')
        if app_headers:
            location = LOCATION(app_headers)
            if location:
                headers = list(app_headers) + list(forget_headers)
                return HTTPFound(headers=headers)
        query = parse_dict_querystring(environ)
        hidden = []
        for key, val in query.items():
            hidden.append(HIDDEN_PRE_LINE % (f'_{key}_', val))
        logger.info('hidden: %s', hidden)
        form = self.formbody or _DEFAULT_FORM
        form = form % '\n'.join(hidden)
        if self.formcallable is not None:
            form = self.formcallable(environ)

        def auth_form(environ, start_response):
            content_length = CONTENT_LENGTH.tuples(str(len(form)))
            content_type = CONTENT_TYPE.tuples('text/html')
            headers = content_length + content_type + forget_headers
            start_response('200 OK', headers)
            return [form]
        return auth_form