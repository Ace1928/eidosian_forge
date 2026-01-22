def _simulate_update(self, resource, timeout, update, wait, **kwargs):
    resource_attributes = update.get('resource_attributes')
    if resource_attributes:
        for k, v in resource_attributes.items():
            resource[k] = v
    return resource