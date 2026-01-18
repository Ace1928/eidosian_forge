from importlib import import_module
import inspect
import io
import token
import tokenize
import traceback
from sphinx.ext.autodoc import ClassLevelDocumenter
from sphinx.util import logging
from traits.has_traits import MetaHasTraits
from traits.trait_type import TraitType
from traits.traits import generic_trait
def trait_definition(*, cls, trait_name):
    """ Retrieve the portion of the source defining a Trait attribute.

    For example, given a class::

        class MyModel(HasStrictTraits)
            foo = List(Int, [1, 2, 3])

    ``trait_definition(cls=MyModel, trait_name="foo")`` returns
    ``"List(Int, [1, 2, 3])"``.

    Parameters
    ----------
    cls : MetaHasTraits
        Class being documented.
    trait_name : str
        Name of the trait being documented.

    Returns
    -------
    str
        The portion of the source containing the trait definition. For
        example, for a class trait defined as ``"my_trait = Float(3.5)"``,
        the returned string will contain ``"Float(3.5)"``.

    Raises
    ------
    ValueError
        If *trait_name* doesn't appear as a class-level variable in the
        source.
    """
    source = inspect.getsource(cls)
    string_io = io.StringIO(source)
    tokens = tokenize.generate_tokens(string_io.readline)
    trait_found = False
    name_found = False
    while not trait_found:
        item = next(tokens, None)
        if item is None:
            break
        if name_found and item[:2] == (token.OP, '='):
            trait_found = True
            continue
        if item[:2] == (token.NAME, trait_name):
            name_found = True
    if not trait_found:
        raise ValueError('No trait definition for {!r} found in {!r}'.format(trait_name, cls))
    definition_tokens = _get_definition_tokens(tokens)
    definition = tokenize.untokenize(definition_tokens).strip()
    return definition