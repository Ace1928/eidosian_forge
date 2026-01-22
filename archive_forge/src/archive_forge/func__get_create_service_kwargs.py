import copy
from docker.errors import create_unexpected_kwargs_error, InvalidArgument
from docker.types import TaskTemplate, ContainerSpec, Placement, ServiceMode
from .resource import Model, Collection
def _get_create_service_kwargs(func_name, kwargs):
    create_kwargs = {}
    for key in copy.copy(kwargs):
        if key in CREATE_SERVICE_KWARGS:
            create_kwargs[key] = kwargs.pop(key)
    container_spec_kwargs = {}
    for key in copy.copy(kwargs):
        if key in CONTAINER_SPEC_KWARGS:
            container_spec_kwargs[key] = kwargs.pop(key)
    task_template_kwargs = {}
    for key in copy.copy(kwargs):
        if key in TASK_TEMPLATE_KWARGS:
            task_template_kwargs[key] = kwargs.pop(key)
    if 'container_labels' in kwargs:
        container_spec_kwargs['labels'] = kwargs.pop('container_labels')
    placement = {}
    for key in copy.copy(kwargs):
        if key in PLACEMENT_KWARGS:
            placement[key] = kwargs.pop(key)
    placement = Placement(**placement)
    task_template_kwargs['placement'] = placement
    if 'log_driver' in kwargs:
        task_template_kwargs['log_driver'] = {'Name': kwargs.pop('log_driver'), 'Options': kwargs.pop('log_driver_options', {})}
    if func_name == 'update':
        if 'force_update' in kwargs:
            task_template_kwargs['force_update'] = kwargs.pop('force_update')
        fetch_current_spec = kwargs.pop('fetch_current_spec', True)
        create_kwargs['fetch_current_spec'] = fetch_current_spec
    if kwargs:
        raise create_unexpected_kwargs_error(func_name, kwargs)
    container_spec = ContainerSpec(**container_spec_kwargs)
    task_template_kwargs['container_spec'] = container_spec
    create_kwargs['task_template'] = TaskTemplate(**task_template_kwargs)
    return create_kwargs