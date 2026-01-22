class TestApplication(object):
    """
    A test WSGI application, that prints out all the environmental
    variables, and if you add ``?error=t`` to the URL it will
    deliberately throw an exception.
    """

    def __init__(self, global_conf=None, text=False):
        self.global_conf = global_conf
        self.text = text

    def __call__(self, environ, start_response):
        if environ.get('QUERY_STRING', '').find('error=') >= 0:
            assert 0, 'Here is your error report, ordered and delivered'
        if self.text:
            page_template = text_page_template
            row_template = text_row_template
            content_type = 'text/plain; charset=utf8'
        else:
            page_template = html_page_template
            row_template = html_row_template
            content_type = 'text/html; charset=utf8'
        keys = sorted(environ.keys())
        rows = []
        for key in keys:
            data = {'key': key}
            value = environ[key]
            data['value'] = value
            try:
                value = repr(value)
            except Exception as e:
                value = 'Cannot use repr(): %s' % e
            data['value_repr'] = value
            data['value_literal'] = make_literal(value)
            row = row_template % data
            rows.append(row)
        rows = ''.join(rows)
        page = page_template % {'environ': rows}
        if isinstance(page, str):
            page = page.encode('utf8')
        headers = [('Content-type', content_type)]
        start_response('200 OK', headers)
        return [page]