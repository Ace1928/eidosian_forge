import warnings
class FeedParserDict(dict):
    keymap = {'channel': 'feed', 'items': 'entries', 'guid': 'id', 'date': 'updated', 'date_parsed': 'updated_parsed', 'description': ['summary', 'subtitle'], 'description_detail': ['summary_detail', 'subtitle_detail'], 'url': ['href'], 'modified': 'updated', 'modified_parsed': 'updated_parsed', 'issued': 'published', 'issued_parsed': 'published_parsed', 'copyright': 'rights', 'copyright_detail': 'rights_detail', 'tagline': 'subtitle', 'tagline_detail': 'subtitle_detail'}

    def __getitem__(self, key):
        """
        :return: A :class:`FeedParserDict`.
        """
        if key == 'category':
            try:
                return dict.__getitem__(self, 'tags')[0]['term']
            except IndexError:
                raise KeyError("object doesn't have key 'category'")
        elif key == 'enclosures':
            norel = lambda link: FeedParserDict([(name, value) for name, value in link.items() if name != 'rel'])
            return [norel(link) for link in dict.__getitem__(self, 'links') if link['rel'] == 'enclosure']
        elif key == 'license':
            for link in dict.__getitem__(self, 'links'):
                if link['rel'] == 'license' and 'href' in link:
                    return link['href']
        elif key == 'updated':
            if not dict.__contains__(self, 'updated') and dict.__contains__(self, 'published'):
                warnings.warn("To avoid breaking existing software while fixing issue 310, a temporary mapping has been created from `updated` to `published` if `updated` doesn't exist. This fallback will be removed in a future version of feedparser.", DeprecationWarning)
                return dict.__getitem__(self, 'published')
            return dict.__getitem__(self, 'updated')
        elif key == 'updated_parsed':
            if not dict.__contains__(self, 'updated_parsed') and dict.__contains__(self, 'published_parsed'):
                warnings.warn("To avoid breaking existing software while fixing issue 310, a temporary mapping has been created from `updated_parsed` to `published_parsed` if `updated_parsed` doesn't exist. This fallback will be removed in a future version of feedparser.", DeprecationWarning)
                return dict.__getitem__(self, 'published_parsed')
            return dict.__getitem__(self, 'updated_parsed')
        else:
            realkey = self.keymap.get(key, key)
            if isinstance(realkey, list):
                for k in realkey:
                    if dict.__contains__(self, k):
                        return dict.__getitem__(self, k)
            elif dict.__contains__(self, realkey):
                return dict.__getitem__(self, realkey)
        return dict.__getitem__(self, key)

    def __contains__(self, key):
        if key in ('updated', 'updated_parsed'):
            return dict.__contains__(self, key)
        try:
            self.__getitem__(key)
        except KeyError:
            return False
        else:
            return True
    has_key = __contains__

    def get(self, key, default=None):
        """
        :return: A :class:`FeedParserDict`.
        """
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __setitem__(self, key, value):
        key = self.keymap.get(key, key)
        if isinstance(key, list):
            key = key[0]
        return dict.__setitem__(self, key, value)

    def setdefault(self, k, default):
        if k not in self:
            self[k] = default
            return default
        return self[k]

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError("object has no attribute '%s'" % key)

    def __hash__(self):
        return id(self)