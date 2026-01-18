from .trait_errors import TraitError
def trait_factory(trait):
    """ Returns a trait created from a TraitFactory instance """
    global _trait_factory_instances
    tid = id(trait)
    if tid not in _trait_factory_instances:
        _trait_factory_instances[tid] = trait()
    return _trait_factory_instances[tid]