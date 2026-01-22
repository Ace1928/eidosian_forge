from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def _process_export_per_selector(self, selector, schema, param, log, export_path, process, schema_invt):
    url = None
    export_urls = schema['urls']
    if 'adom' in param and (not export_urls[0].endswith('{adom}')):
        if param['adom'] == 'global':
            for _url in export_urls:
                if '/global/' in _url and '/adom/{adom}/' not in _url:
                    url = _url
                    break
        else:
            for _url in export_urls:
                if '/adom/{adom}/' in _url:
                    url = _url
                    break
    if not url:
        url = export_urls[0]
    _param_applied = list()
    for _param_key in param:
        _param_value = param[_param_key]
        if _param_key == 'adom' and _param_value.lower() == 'global':
            continue
        token_hint = '/%s/{%s}' % (_param_key, _param_key)
        token = '/%s/%s' % (_param_key, _param_value)
        if token_hint in url:
            _param_applied.append(_param_key)
        url = url.replace(token_hint, token)
    for _param_key in param:
        if _param_key in _param_applied:
            continue
        if _param_key == 'adom' and _param_value.lower() == 'global':
            continue
        token_hint = '{%s}' % _param_key
        token = param[_param_key]
        url = url.replace(token_hint, token)
    tokens = url.split('/')
    if tokens[-1].startswith('{') and tokens[-1].endswith('}'):
        new_url = ''
        for token in tokens[:-1]:
            new_url += '/%s' % token
        new_url = new_url.replace('//', '/')
        url = new_url
    unresolved_parameter = False
    tokens = url.split('/')
    for token in tokens:
        if token.startswith('{') and token.endswith('}'):
            unresolved_parameter = True
            break
    log.write('[%s]exporting: %s\n' % (process, selector))
    log.write('\turl: %s\n' % url)
    if unresolved_parameter:
        log.write('\t unknown parameter, skipped!\n')
        return
    response = self.conn.send_request('get', [{'url': url}])
    self._process_export_response(selector, response, schema_invt, log, export_path, param, schema['params'])