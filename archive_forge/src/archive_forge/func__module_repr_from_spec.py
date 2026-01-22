def _module_repr_from_spec(spec):
    """Return the repr to use for the module."""
    name = '?' if spec.name is None else spec.name
    if spec.origin is None:
        if spec.loader is None:
            return '<module {!r}>'.format(name)
        else:
            return '<module {!r} ({!r})>'.format(name, spec.loader)
    elif spec.has_location:
        return '<module {!r} from {!r}>'.format(name, spec.origin)
    else:
        return '<module {!r} ({})>'.format(spec.name, spec.origin)