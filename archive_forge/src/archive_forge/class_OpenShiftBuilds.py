from __future__ import (absolute_import, division, print_function)
from datetime import datetime, timezone, timedelta
import traceback
import time
from ansible.module_utils._text import to_native
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
class OpenShiftBuilds(AnsibleOpenshiftModule):

    def __init__(self, **kwargs):
        super(OpenShiftBuilds, self).__init__(**kwargs)

    def get_build_config(self, name, namespace):
        params = dict(kind='BuildConfig', api_version='build.openshift.io/v1', name=name, namespace=namespace)
        result = self.kubernetes_facts(**params)
        return result['resources']

    def clone_build(self, name, namespace, request):
        try:
            result = self.request(method='POST', path='/apis/build.openshift.io/v1/namespaces/{namespace}/builds/{name}/clone'.format(namespace=namespace, name=name), body=request, content_type='application/json')
            return result.to_dict()
        except DynamicApiError as exc:
            msg = 'Failed to clone Build %s/%s due to: %s' % (namespace, name, exc.body)
            self.fail_json(msg=msg, status=exc.status, reason=exc.reason)
        except Exception as e:
            msg = 'Failed to clone Build %s/%s due to: %s' % (namespace, name, to_native(e))
            self.fail_json(msg=msg, error=to_native(e), exception=e)

    def instantiate_build_config(self, name, namespace, request):
        try:
            result = self.request(method='POST', path='/apis/build.openshift.io/v1/namespaces/{namespace}/buildconfigs/{name}/instantiate'.format(namespace=namespace, name=name), body=request, content_type='application/json')
            return result.to_dict()
        except DynamicApiError as exc:
            msg = 'Failed to instantiate BuildConfig %s/%s due to: %s' % (namespace, name, exc.body)
            self.fail_json(msg=msg, status=exc.status, reason=exc.reason)
        except Exception as e:
            msg = 'Failed to instantiate BuildConfig %s/%s due to: %s' % (namespace, name, to_native(e))
            self.fail_json(msg=msg, error=to_native(e), exception=e)

    def start_build(self):
        result = None
        name = self.params.get('build_config_name')
        if not name:
            name = self.params.get('build_name')
        build_request = {'kind': 'BuildRequest', 'apiVersion': 'build.openshift.io/v1', 'metadata': {'name': name}, 'triggeredBy': [{'message': 'Manually triggered'}]}
        incremental = self.params.get('incremental')
        if incremental is not None:
            build_request.update({'sourceStrategyOptions': {'incremental': incremental}})
        if self.params.get('env_vars'):
            build_request.update({'env': self.params.get('env_vars')})
        if self.params.get('build_args'):
            build_request.update({'dockerStrategyOptions': {'buildArgs': self.params.get('build_args')}})
        no_cache = self.params.get('no_cache')
        if no_cache is not None:
            build_request.update({'dockerStrategyOptions': {'noCache': no_cache}})
        if self.params.get('commit'):
            build_request.update({'revision': {'git': {'commit': self.params.get('commit')}}})
        if self.params.get('build_config_name'):
            result = self.instantiate_build_config(name=self.params.get('build_config_name'), namespace=self.params.get('namespace'), request=build_request)
        else:
            result = self.clone_build(name=self.params.get('build_name'), namespace=self.params.get('namespace'), request=build_request)
        if result and self.params.get('wait'):
            start = datetime.now()

            def _total_wait_time():
                return (datetime.now() - start).seconds
            wait_timeout = self.params.get('wait_timeout')
            wait_sleep = self.params.get('wait_sleep')
            last_status_phase = None
            while _total_wait_time() < wait_timeout:
                params = dict(kind=result['kind'], api_version=result['apiVersion'], name=result['metadata']['name'], namespace=result['metadata']['namespace'])
                facts = self.kubernetes_facts(**params)
                if len(facts['resources']) > 0:
                    last_status_phase = facts['resources'][0]['status']['phase']
                    if last_status_phase == 'Complete':
                        result = facts['resources'][0]
                        break
                    elif last_status_phase in ('Cancelled', 'Error', 'Failed'):
                        self.fail_json(msg='Unexpected status for Build %s/%s: %s' % (result['metadata']['name'], result['metadata']['namespace'], last_status_phase))
                time.sleep(wait_sleep)
            if last_status_phase != 'Complete':
                name = result['metadata']['name']
                namespace = result['metadata']['namespace']
                msg = 'Build %s/%s has not complete after %d second(s),current status is %s' % (namespace, name, wait_timeout, last_status_phase)
                self.fail_json(msg=msg)
        result = [result] if result else []
        self.exit_json(changed=True, builds=result)

    def cancel_build(self, restart):
        kind = 'Build'
        api_version = 'build.openshift.io/v1'
        namespace = self.params.get('namespace')
        phases = ['new', 'pending', 'running']
        build_phases = self.params.get('build_phases', [])
        if build_phases:
            phases = [p.lower() for p in build_phases]
        names = []
        if self.params.get('build_name'):
            names.append(self.params.get('build_name'))
        else:
            build_config = self.params.get('build_config_name')
            params = dict(kind=kind, api_version=api_version, namespace=namespace)
            resources = self.kubernetes_facts(**params).get('resources', [])

            def _filter_builds(build):
                config = build['metadata'].get('labels', {}).get('openshift.io/build-config.name')
                return build_config is None or (build_config is not None and config in build_config)
            for item in list(filter(_filter_builds, resources)):
                name = item['metadata']['name']
                if name not in names:
                    names.append(name)
        if len(names) == 0:
            self.exit_json(changed=False, msg='No Build found from namespace %s' % namespace)
        warning = []
        builds_to_cancel = []
        for name in names:
            params = dict(kind=kind, api_version=api_version, name=name, namespace=namespace)
            resource = self.kubernetes_facts(**params).get('resources', [])
            if len(resource) == 0:
                warning.append('Build %s/%s not found' % (namespace, name))
                continue
            resource = resource[0]
            phase = resource['status'].get('phase').lower()
            if phase in phases:
                builds_to_cancel.append(resource)
            else:
                warning.append('build %s/%s is not in expected phase, found %s' % (namespace, name, phase))
        changed = False
        result = []
        for build in builds_to_cancel:
            build['status']['cancelled'] = True
            name = build['metadata']['name']
            changed = True
            try:
                content_type = 'application/json'
                cancelled_build = self.request('PUT', '/apis/build.openshift.io/v1/namespaces/{0}/builds/{1}'.format(namespace, name), body=build, content_type=content_type).to_dict()
                result.append(cancelled_build)
            except DynamicApiError as exc:
                self.fail_json(msg='Failed to cancel Build %s/%s due to: %s' % (namespace, name, exc), reason=exc.reason, status=exc.status)
            except Exception as e:
                self.fail_json(msg='Failed to cancel Build %s/%s due to: %s' % (namespace, name, e))

        def _wait_until_cancelled(build, wait_timeout, wait_sleep):
            start = datetime.now()
            last_phase = None
            name = build['metadata']['name']
            while (datetime.now() - start).seconds < wait_timeout:
                params = dict(kind=kind, api_version=api_version, name=name, namespace=namespace)
                resource = self.kubernetes_facts(**params).get('resources', [])
                if len(resource) == 0:
                    return (None, 'Build %s/%s not found' % (namespace, name))
                resource = resource[0]
                last_phase = resource['status']['phase']
                if last_phase == 'Cancelled':
                    return (resource, None)
                time.sleep(wait_sleep)
            return (None, 'Build %s/%s is not cancelled as expected, current state is %s' % (namespace, name, last_phase))
        if result and self.params.get('wait'):
            wait_timeout = self.params.get('wait_timeout')
            wait_sleep = self.params.get('wait_sleep')
            wait_result = []
            for build in result:
                ret, err = _wait_until_cancelled(build, wait_timeout, wait_sleep)
                if err:
                    self.exit_json(msg=err)
                wait_result.append(ret)
            result = wait_result
        if restart:
            self.start_build()
        self.exit_json(builds=result, changed=changed)

    def execute_module(self):
        state = self.params.get('state')
        if state == 'started':
            self.start_build()
        else:
            restart = bool(state == 'restarted')
            self.cancel_build(restart=restart)