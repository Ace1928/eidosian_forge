from .schema import rest_translation
def uri_shape(uri):
    import re
    kwid_map = dict(zip(uri.split('/')[1::2], uri.split('/')[2::2]))
    shapes = {}
    for kw in kwid_map:
        seps = kwid_map[kw]
        for char in re.findall('[a-zA-Z0-9]', seps):
            seps = seps.replace(char, '')
            chunks = []
            p = '|'.join(seps)
            s = re.split(p, kwid_map[kw]) if p != '' else kwid_map[kw]
            for chunk in s:
                try:
                    float(chunk)
                    chunk = '*'
                except Exception:
                    pass
                chunks.append(chunk)
            shapes[kw] = '?'.join(chunks)
    return make_uri(shapes)