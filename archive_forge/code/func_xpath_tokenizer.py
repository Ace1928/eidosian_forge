import re
def xpath_tokenizer(pattern, namespaces=None, with_prefixes=True):
    default_namespace = namespaces.get(None) or namespaces.get('') if namespaces else None
    parsing_attribute = False
    for token in xpath_tokenizer_re.findall(pattern):
        ttype, tag = token
        if tag and tag[0] != '{':
            if ':' in tag and with_prefixes:
                prefix, uri = tag.split(':', 1)
                try:
                    if not namespaces:
                        raise KeyError
                    yield (ttype, '{%s}%s' % (namespaces[prefix], uri))
                except KeyError:
                    raise SyntaxError('prefix %r not found in prefix map' % prefix)
            elif default_namespace and (not parsing_attribute):
                yield (ttype, '{%s}%s' % (default_namespace, tag))
            else:
                yield token
            parsing_attribute = False
        else:
            yield token
            parsing_attribute = ttype == '@'