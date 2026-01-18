from .. import auth, errors, utils
from ..types import ServiceMode
@utils.minimum_version('1.24')
@utils.check_resource('service')
def update_service(self, service, version, task_template=None, name=None, labels=None, mode=None, update_config=None, networks=None, endpoint_config=None, endpoint_spec=None, fetch_current_spec=False, rollback_config=None):
    """
        Update a service.

        Args:
            service (string): A service identifier (either its name or service
                ID).
            version (int): The version number of the service object being
                updated. This is required to avoid conflicting writes.
            task_template (TaskTemplate): Specification of the updated task to
                start as part of the service.
            name (string): New name for the service. Optional.
            labels (dict): A map of labels to associate with the service.
                Optional.
            mode (ServiceMode): Scheduling mode for the service (replicated
                or global). Defaults to replicated.
            update_config (UpdateConfig): Specification for the update strategy
                of the service. Default: ``None``.
            rollback_config (RollbackConfig): Specification for the rollback
                strategy of the service. Default: ``None``
            networks (:py:class:`list`): List of network names or IDs or
                :py:class:`~docker.types.NetworkAttachmentConfig` to attach the
                service to. Default: ``None``.
            endpoint_spec (EndpointSpec): Properties that can be configured to
                access and load balance a service. Default: ``None``.
            fetch_current_spec (boolean): Use the undefined settings from the
                current specification of the service. Default: ``False``

        Returns:
            A dictionary containing a ``Warnings`` key.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    _check_api_features(self._version, task_template, update_config, endpoint_spec, rollback_config)
    if fetch_current_spec:
        inspect_defaults = True
        if utils.version_lt(self._version, '1.29'):
            inspect_defaults = None
        current = self.inspect_service(service, insert_defaults=inspect_defaults)['Spec']
    else:
        current = {}
    url = self._url('/services/{0}/update', service)
    data = {}
    headers = {}
    data['Name'] = current.get('Name') if name is None else name
    data['Labels'] = current.get('Labels') if labels is None else labels
    if mode is not None:
        if not isinstance(mode, dict):
            mode = ServiceMode(mode)
        data['Mode'] = mode
    else:
        data['Mode'] = current.get('Mode')
    data['TaskTemplate'] = _merge_task_template(current.get('TaskTemplate', {}), task_template)
    container_spec = data['TaskTemplate'].get('ContainerSpec', {})
    image = container_spec.get('Image', None)
    if image is not None:
        registry, repo_name = auth.resolve_repository_name(image)
        auth_header = auth.get_config_header(self, registry)
        if auth_header:
            headers['X-Registry-Auth'] = auth_header
    if update_config is not None:
        data['UpdateConfig'] = update_config
    else:
        data['UpdateConfig'] = current.get('UpdateConfig')
    if rollback_config is not None:
        data['RollbackConfig'] = rollback_config
    else:
        data['RollbackConfig'] = current.get('RollbackConfig')
    if networks is not None:
        converted_networks = utils.convert_service_networks(networks)
        if utils.version_lt(self._version, '1.25'):
            data['Networks'] = converted_networks
        else:
            data['TaskTemplate']['Networks'] = converted_networks
    elif utils.version_lt(self._version, '1.25'):
        data['Networks'] = current.get('Networks')
    elif data['TaskTemplate'].get('Networks') is None:
        current_task_template = current.get('TaskTemplate', {})
        current_networks = current_task_template.get('Networks')
        if current_networks is None:
            current_networks = current.get('Networks')
        if current_networks is not None:
            data['TaskTemplate']['Networks'] = current_networks
    if endpoint_spec is not None:
        data['EndpointSpec'] = endpoint_spec
    else:
        data['EndpointSpec'] = current.get('EndpointSpec')
    resp = self._post_json(url, data=data, params={'version': version}, headers=headers)
    return self._result(resp, json=True)