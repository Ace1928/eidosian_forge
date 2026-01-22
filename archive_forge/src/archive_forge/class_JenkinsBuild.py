from __future__ import absolute_import, division, print_function
import traceback
from time import sleep
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
class JenkinsBuild:

    def __init__(self, module):
        self.module = module
        self.name = module.params.get('name')
        self.password = module.params.get('password')
        self.args = module.params.get('args')
        self.state = module.params.get('state')
        self.token = module.params.get('token')
        self.user = module.params.get('user')
        self.jenkins_url = module.params.get('url')
        self.build_number = module.params.get('build_number')
        self.detach = module.params.get('detach')
        self.time_between_checks = module.params.get('time_between_checks')
        self.server = self.get_jenkins_connection()
        self.result = {'changed': False, 'url': self.jenkins_url, 'name': self.name, 'user': self.user, 'state': self.state}
        self.EXCL_STATE = 'excluded state'

    def get_jenkins_connection(self):
        try:
            if self.user and self.password:
                return jenkins.Jenkins(self.jenkins_url, self.user, self.password)
            elif self.user and self.token:
                return jenkins.Jenkins(self.jenkins_url, self.user, self.token)
            elif self.user and (not (self.password or self.token)):
                return jenkins.Jenkins(self.jenkins_url, self.user)
            else:
                return jenkins.Jenkins(self.jenkins_url)
        except Exception as e:
            self.module.fail_json(msg='Unable to connect to Jenkins server, %s' % to_native(e))

    def get_next_build(self):
        try:
            build_number = self.server.get_job_info(self.name)['nextBuildNumber']
        except Exception as e:
            self.module.fail_json(msg='Unable to get job info from Jenkins server, %s' % to_native(e), exception=traceback.format_exc())
        return build_number

    def get_build_status(self):
        try:
            response = self.server.get_build_info(self.name, self.build_number)
            return response
        except jenkins.JenkinsException as e:
            response = {}
            response['result'] = 'ABSENT'
            return response
        except Exception as e:
            self.module.fail_json(msg='Unable to fetch build information, %s' % to_native(e), exception=traceback.format_exc())

    def present_build(self):
        self.build_number = self.get_next_build()
        try:
            if self.args is None:
                self.server.build_job(self.name)
            else:
                self.server.build_job(self.name, self.args)
        except Exception as e:
            self.module.fail_json(msg='Unable to create build for %s: %s' % (self.jenkins_url, to_native(e)), exception=traceback.format_exc())

    def stopped_build(self):
        build_info = None
        try:
            build_info = self.server.get_build_info(self.name, self.build_number)
            if build_info['building'] is True:
                self.server.stop_build(self.name, self.build_number)
        except Exception as e:
            self.module.fail_json(msg='Unable to stop build for %s: %s' % (self.jenkins_url, to_native(e)), exception=traceback.format_exc())
        else:
            if build_info['building'] is False:
                self.module.exit_json(**self.result)

    def absent_build(self):
        try:
            self.server.delete_build(self.name, self.build_number)
        except Exception as e:
            self.module.fail_json(msg='Unable to delete build for %s: %s' % (self.jenkins_url, to_native(e)), exception=traceback.format_exc())

    def get_result(self):
        result = self.result
        build_status = self.get_build_status()
        if build_status['result'] is None:
            if self.detach:
                result['changed'] = True
                result['build_info'] = build_status
                return result
            sleep(self.time_between_checks)
            self.get_result()
        elif self.state == 'stopped' and build_status['result'] == 'ABORTED':
            result['changed'] = True
            result['build_info'] = build_status
        elif self.state == 'absent' and build_status['result'] == 'ABSENT':
            result['changed'] = True
            result['build_info'] = build_status
        elif self.state != 'absent' and build_status['result'] == 'SUCCESS':
            result['changed'] = True
            result['build_info'] = build_status
        else:
            result['failed'] = True
            result['build_info'] = build_status
        return result