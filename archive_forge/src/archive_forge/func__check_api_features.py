from .. import auth, errors, utils
from ..types import ServiceMode
def _check_api_features(version, task_template, update_config, endpoint_spec, rollback_config):

    def raise_version_error(param, min_version):
        raise errors.InvalidVersion('{} is not supported in API version < {}'.format(param, min_version))
    if update_config is not None:
        if utils.version_lt(version, '1.25'):
            if 'MaxFailureRatio' in update_config:
                raise_version_error('UpdateConfig.max_failure_ratio', '1.25')
            if 'Monitor' in update_config:
                raise_version_error('UpdateConfig.monitor', '1.25')
        if utils.version_lt(version, '1.28'):
            if update_config.get('FailureAction') == 'rollback':
                raise_version_error('UpdateConfig.failure_action rollback', '1.28')
        if utils.version_lt(version, '1.29'):
            if 'Order' in update_config:
                raise_version_error('UpdateConfig.order', '1.29')
    if rollback_config is not None:
        if utils.version_lt(version, '1.28'):
            raise_version_error('rollback_config', '1.28')
        if utils.version_lt(version, '1.29'):
            if 'Order' in update_config:
                raise_version_error('RollbackConfig.order', '1.29')
    if endpoint_spec is not None:
        if utils.version_lt(version, '1.32') and 'Ports' in endpoint_spec:
            if any((p.get('PublishMode') for p in endpoint_spec['Ports'])):
                raise_version_error('EndpointSpec.Ports[].mode', '1.32')
    if task_template is not None:
        if 'ForceUpdate' in task_template and utils.version_lt(version, '1.25'):
            raise_version_error('force_update', '1.25')
        if task_template.get('Placement'):
            if utils.version_lt(version, '1.30'):
                if task_template['Placement'].get('Platforms'):
                    raise_version_error('Placement.platforms', '1.30')
            if utils.version_lt(version, '1.27'):
                if task_template['Placement'].get('Preferences'):
                    raise_version_error('Placement.preferences', '1.27')
        if task_template.get('ContainerSpec'):
            container_spec = task_template.get('ContainerSpec')
            if utils.version_lt(version, '1.25'):
                if container_spec.get('TTY'):
                    raise_version_error('ContainerSpec.tty', '1.25')
                if container_spec.get('Hostname') is not None:
                    raise_version_error('ContainerSpec.hostname', '1.25')
                if container_spec.get('Hosts') is not None:
                    raise_version_error('ContainerSpec.hosts', '1.25')
                if container_spec.get('Groups') is not None:
                    raise_version_error('ContainerSpec.groups', '1.25')
                if container_spec.get('DNSConfig') is not None:
                    raise_version_error('ContainerSpec.dns_config', '1.25')
                if container_spec.get('Healthcheck') is not None:
                    raise_version_error('ContainerSpec.healthcheck', '1.25')
            if utils.version_lt(version, '1.28'):
                if container_spec.get('ReadOnly') is not None:
                    raise_version_error('ContainerSpec.dns_config', '1.28')
                if container_spec.get('StopSignal') is not None:
                    raise_version_error('ContainerSpec.stop_signal', '1.28')
            if utils.version_lt(version, '1.30'):
                if container_spec.get('Configs') is not None:
                    raise_version_error('ContainerSpec.configs', '1.30')
                if container_spec.get('Privileges') is not None:
                    raise_version_error('ContainerSpec.privileges', '1.30')
            if utils.version_lt(version, '1.35'):
                if container_spec.get('Isolation') is not None:
                    raise_version_error('ContainerSpec.isolation', '1.35')
            if utils.version_lt(version, '1.38'):
                if container_spec.get('Init') is not None:
                    raise_version_error('ContainerSpec.init', '1.38')
        if task_template.get('Resources'):
            if utils.version_lt(version, '1.32'):
                if task_template['Resources'].get('GenericResources'):
                    raise_version_error('Resources.generic_resources', '1.32')