def _get_deep(properties, *keys):
    """Get a final key among a list of keys (each with its own sub-dict)."""
    for key in keys:
        properties = properties[key]
    return properties