from collections import abc
class KeyValueView(object):
    """A Key-Value Text View

    This view performs an advanced serialization of a model
    into text by following the following set of rules:

    key : text
        key = text

    rootkey : Mapping
        ::

            rootkey =
              serialize(key, value)

    key : Sequence
        ::

            key =
              serialize(item)

    :param str indent_str: the string used to represent one "indent"
    :param str key_sep: the separator to use between keys and values
    :param str dict_sep: the separator to use after a dictionary root key
    :param str list_sep: the separator to use after a list root key
    :param str anon_dict: the "key" to use when there is a dict in a list
                          (does not automatically use the dict separator)
    :param before_dict: content to place on the line(s) before the a dict
                        root key (use None to avoid inserting an extra line)
    :type before_dict: str or None
    :param before_list: content to place on the line(s) before the a list
                        root key (use None to avoid inserting an extra line)
    :type before_list: str or None
    """

    def __init__(self, indent_str='  ', key_sep=' = ', dict_sep=' = ', list_sep=' = ', anon_dict='[dict]', before_dict=None, before_list=None):
        self.indent_str = indent_str
        self.key_sep = key_sep
        self.dict_sep = dict_sep
        self.list_sep = list_sep
        self.anon_dict = anon_dict
        self.before_dict = before_dict
        self.before_list = before_list

    def __call__(self, model):

        def serialize(root, rootkey, indent):
            res = []
            if rootkey is not None:
                res.append(self.indent_str * indent + str(rootkey))
            if isinstance(root, abc.Mapping):
                if rootkey is None and indent > 0:
                    res.append(self.indent_str * indent + self.anon_dict)
                elif rootkey is not None:
                    res[0] += self.dict_sep
                    if self.before_dict is not None:
                        res.insert(0, self.before_dict)
                for key in sorted(root, key=str):
                    res.extend(serialize(root[key], key, indent + 1))
            elif isinstance(root, abc.Sequence) and (not isinstance(root, str)):
                if rootkey is not None:
                    res[0] += self.list_sep
                    if self.before_list is not None:
                        res.insert(0, self.before_list)
                for val in sorted(root, key=str):
                    res.extend(serialize(val, None, indent + 1))
            else:
                str_root = str(root)
                if '\n' in str_root:
                    if rootkey is not None:
                        res[0] += self.dict_sep
                    list_root = [self.indent_str * (indent + 1) + line for line in str_root.split('\n')]
                    res.extend(list_root)
                else:
                    try:
                        res[0] += self.key_sep + str_root
                    except IndexError:
                        res = [self.indent_str * indent + str_root]
            return res
        return '\n'.join(serialize(model, None, -1))