from sympy.core import S, Basic, Dict, Symbol, Tuple, sympify
from sympy.core.symbol import Str
from sympy.sets import Set, FiniteSet, EmptySet
from sympy.utilities.iterables import iterable
@staticmethod
def _add_morphism_closure(morphisms, morphism, props, add_identities=True, recurse_composites=True):
    """
        Adds a morphism and its attributes to the supplied dictionary
        ``morphisms``.  If ``add_identities`` is True, also adds the
        identity morphisms for the domain and the codomain of
        ``morphism``.
        """
    if not Diagram._set_dict_union(morphisms, morphism, props):
        if isinstance(morphism, IdentityMorphism):
            if props:
                raise ValueError('Instances of IdentityMorphism cannot have properties.')
            return
        if add_identities:
            empty = EmptySet
            id_dom = IdentityMorphism(morphism.domain)
            id_cod = IdentityMorphism(morphism.codomain)
            Diagram._set_dict_union(morphisms, id_dom, empty)
            Diagram._set_dict_union(morphisms, id_cod, empty)
        for existing_morphism, existing_props in list(morphisms.items()):
            new_props = existing_props & props
            if morphism.domain == existing_morphism.codomain:
                left = morphism * existing_morphism
                Diagram._set_dict_union(morphisms, left, new_props)
            if morphism.codomain == existing_morphism.domain:
                right = existing_morphism * morphism
                Diagram._set_dict_union(morphisms, right, new_props)
        if isinstance(morphism, CompositeMorphism) and recurse_composites:
            empty = EmptySet
            for component in morphism.components:
                Diagram._add_morphism_closure(morphisms, component, empty, add_identities)