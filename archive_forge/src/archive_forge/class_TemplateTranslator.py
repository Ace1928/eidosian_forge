from typing import Union, Optional, Mapping, Dict, Tuple, Iterator
from lark import Tree, Transformer
from lark.exceptions import MissingVariableError
class TemplateTranslator:
    """Utility class for translating a collection of patterns
    """

    def __init__(self, translations: Mapping[Template, Template]):
        assert all((isinstance(k, Template) and isinstance(v, Template) for k, v in translations.items()))
        self.translations = translations

    def translate(self, tree: Tree[str]):
        for k, v in self.translations.items():
            tree = translate(k, v, tree)
        return tree