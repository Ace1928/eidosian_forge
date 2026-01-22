class _LooseFeedParser(object):
    contentparams = None

    def __init__(self, baseuri=None, baselang=None, encoding=None, entities=None):
        self.baseuri = baseuri or ''
        self.lang = baselang or None
        self.encoding = encoding or 'utf-8'
        self.entities = entities or {}
        super(_LooseFeedParser, self).__init__()

    @staticmethod
    def _normalize_attributes(kv):
        k = kv[0].lower()
        v = k in ('rel', 'type') and kv[1].lower() or kv[1]
        v = v.replace('&amp;', '&')
        return (k, v)

    def decode_entities(self, element, data):
        data = data.replace('&#60;', '&lt;')
        data = data.replace('&#x3c;', '&lt;')
        data = data.replace('&#x3C;', '&lt;')
        data = data.replace('&#62;', '&gt;')
        data = data.replace('&#x3e;', '&gt;')
        data = data.replace('&#x3E;', '&gt;')
        data = data.replace('&#38;', '&amp;')
        data = data.replace('&#x26;', '&amp;')
        data = data.replace('&#34;', '&quot;')
        data = data.replace('&#x22;', '&quot;')
        data = data.replace('&#39;', '&apos;')
        data = data.replace('&#x27;', '&apos;')
        if not self.contentparams.get('type', 'xml').endswith('xml'):
            data = data.replace('&lt;', '<')
            data = data.replace('&gt;', '>')
            data = data.replace('&amp;', '&')
            data = data.replace('&quot;', '"')
            data = data.replace('&apos;', "'")
            data = data.replace('&#x2f;', '/')
            data = data.replace('&#x2F;', '/')
        return data

    @staticmethod
    def strattrs(attrs):
        return ''.join((' %s="%s"' % (n, v.replace('"', '&quot;')) for n, v in attrs))