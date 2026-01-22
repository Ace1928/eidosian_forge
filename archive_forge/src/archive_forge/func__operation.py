def _operation(name, location, **kwargs):
    return {'operation': name, 'location': location, 'params': dict(**kwargs)}