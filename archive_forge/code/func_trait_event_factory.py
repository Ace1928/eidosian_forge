def trait_event_factory(object, name, old, new):
    """ Adapt the call signature of ctraits call_notifiers to create an event.

    Parameters
    ----------
    object : HasTraits
        Object on which a trait is changed.
    name : str
        Name of the trait.
    old : any
        The old value.
    new : any
        The new value.

    Returns
    -------
    TraitChangeEvent
    """
    return TraitChangeEvent(object=object, name=name, old=old, new=new)