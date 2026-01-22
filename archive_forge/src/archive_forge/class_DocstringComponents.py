import re
import pydoc
from .external.docscrape import NumpyDocString
class DocstringComponents:
    regexp = re.compile('\\n((\\n|.)+)\\n\\s*', re.MULTILINE)

    def __init__(self, comp_dict, strip_whitespace=True):
        """Read entries from a dict, optionally stripping outer whitespace."""
        if strip_whitespace:
            entries = {}
            for key, val in comp_dict.items():
                m = re.match(self.regexp, val)
                if m is None:
                    entries[key] = val
                else:
                    entries[key] = m.group(1)
        else:
            entries = comp_dict.copy()
        self.entries = entries

    def __getattr__(self, attr):
        """Provide dot access to entries for clean raw docstrings."""
        if attr in self.entries:
            return self.entries[attr]
        else:
            try:
                return self.__getattribute__(attr)
            except AttributeError as err:
                if __debug__:
                    raise err
                else:
                    pass

    @classmethod
    def from_nested_components(cls, **kwargs):
        """Add multiple sub-sets of components."""
        return cls(kwargs, strip_whitespace=False)

    @classmethod
    def from_function_params(cls, func):
        """Use the numpydoc parser to extract components from existing func."""
        params = NumpyDocString(pydoc.getdoc(func))['Parameters']
        comp_dict = {}
        for p in params:
            name = p.name
            type = p.type
            desc = '\n    '.join(p.desc)
            comp_dict[name] = f'{name} : {type}\n    {desc}'
        return cls(comp_dict)