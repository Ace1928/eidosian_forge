import unicodedata
def _element_list_(self):
    self._push('element_list')
    self._seq([lambda: self._bind(self._value_, 'v'), self._element_list__s1_, self._sp_, self._element_list__s3_, lambda: self._succeed([self._get('v')] + self._get('vs'))])
    self._pop('element_list')