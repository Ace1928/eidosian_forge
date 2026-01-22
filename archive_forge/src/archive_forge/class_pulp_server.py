from __future__ import absolute_import, division, print_function
import json
import os
from time import sleep
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.urls import url_argument_spec
class pulp_server(object):
    """
    Class to interact with a Pulp server
    """

    def __init__(self, module, pulp_host, repo_type, wait_for_completion=False):
        self.module = module
        self.host = pulp_host
        self.repo_type = repo_type
        self.repo_cache = dict()
        self.wait_for_completion = wait_for_completion

    def check_repo_exists(self, repo_id):
        try:
            self.get_repo_config_by_id(repo_id)
        except IndexError:
            return False
        else:
            return True

    def compare_repo_distributor_config(self, repo_id, **kwargs):
        repo_config = self.get_repo_config_by_id(repo_id)
        for distributor in repo_config['distributors']:
            for key, value in kwargs.items():
                if key not in distributor['config'].keys():
                    return False
                if not distributor['config'][key] == value:
                    return False
        return True

    def compare_repo_importer_config(self, repo_id, **kwargs):
        repo_config = self.get_repo_config_by_id(repo_id)
        for importer in repo_config['importers']:
            for key, value in kwargs.items():
                if value is not None:
                    if key not in importer['config'].keys():
                        return False
                    if not importer['config'][key] == value:
                        return False
        return True

    def create_repo(self, repo_id, relative_url, feed=None, generate_sqlite=False, serve_http=False, serve_https=True, proxy_host=None, proxy_port=None, proxy_username=None, proxy_password=None, repoview=False, ssl_ca_cert=None, ssl_client_cert=None, ssl_client_key=None, add_export_distributor=False):
        url = '%s/pulp/api/v2/repositories/' % self.host
        data = dict()
        data['id'] = repo_id
        data['distributors'] = []
        if self.repo_type == 'rpm':
            yum_distributor = dict()
            yum_distributor['distributor_id'] = 'yum_distributor'
            yum_distributor['distributor_type_id'] = 'yum_distributor'
            yum_distributor['auto_publish'] = True
            yum_distributor['distributor_config'] = dict()
            yum_distributor['distributor_config']['http'] = serve_http
            yum_distributor['distributor_config']['https'] = serve_https
            yum_distributor['distributor_config']['relative_url'] = relative_url
            yum_distributor['distributor_config']['repoview'] = repoview
            yum_distributor['distributor_config']['generate_sqlite'] = generate_sqlite or repoview
            data['distributors'].append(yum_distributor)
            if add_export_distributor:
                export_distributor = dict()
                export_distributor['distributor_id'] = 'export_distributor'
                export_distributor['distributor_type_id'] = 'export_distributor'
                export_distributor['auto_publish'] = False
                export_distributor['distributor_config'] = dict()
                export_distributor['distributor_config']['http'] = serve_http
                export_distributor['distributor_config']['https'] = serve_https
                export_distributor['distributor_config']['relative_url'] = relative_url
                export_distributor['distributor_config']['repoview'] = repoview
                export_distributor['distributor_config']['generate_sqlite'] = generate_sqlite or repoview
                data['distributors'].append(export_distributor)
            data['importer_type_id'] = 'yum_importer'
            data['importer_config'] = dict()
            if feed:
                data['importer_config']['feed'] = feed
            if proxy_host:
                data['importer_config']['proxy_host'] = proxy_host
            if proxy_port:
                data['importer_config']['proxy_port'] = proxy_port
            if proxy_username:
                data['importer_config']['proxy_username'] = proxy_username
            if proxy_password:
                data['importer_config']['proxy_password'] = proxy_password
            if ssl_ca_cert:
                data['importer_config']['ssl_ca_cert'] = ssl_ca_cert
            if ssl_client_cert:
                data['importer_config']['ssl_client_cert'] = ssl_client_cert
            if ssl_client_key:
                data['importer_config']['ssl_client_key'] = ssl_client_key
            data['notes'] = {'_repo-type': 'rpm-repo'}
        response, info = fetch_url(self.module, url, data=json.dumps(data), method='POST')
        if info['status'] != 201:
            self.module.fail_json(msg='Failed to create repo.', status_code=info['status'], response=info['msg'], url=url)
        else:
            return True

    def delete_repo(self, repo_id):
        url = '%s/pulp/api/v2/repositories/%s/' % (self.host, repo_id)
        response, info = fetch_url(self.module, url, data='', method='DELETE')
        if info['status'] != 202:
            self.module.fail_json(msg='Failed to delete repo.', status_code=info['status'], response=info['msg'], url=url)
        if self.wait_for_completion:
            self.verify_tasks_completed(json.load(response))
        return True

    def get_repo_config_by_id(self, repo_id):
        if repo_id not in self.repo_cache.keys():
            repo_array = [x for x in self.repo_list if x['id'] == repo_id]
            self.repo_cache[repo_id] = repo_array[0]
        return self.repo_cache[repo_id]

    def publish_repo(self, repo_id, publish_distributor):
        url = '%s/pulp/api/v2/repositories/%s/actions/publish/' % (self.host, repo_id)
        if publish_distributor is None:
            repo_config = self.get_repo_config_by_id(repo_id)
            for distributor in repo_config['distributors']:
                data = dict()
                data['id'] = distributor['id']
                response, info = fetch_url(self.module, url, data=json.dumps(data), method='POST')
                if info['status'] != 202:
                    self.module.fail_json(msg='Failed to publish the repo.', status_code=info['status'], response=info['msg'], url=url, distributor=distributor['id'])
        else:
            data = dict()
            data['id'] = publish_distributor
            response, info = fetch_url(self.module, url, data=json.dumps(data), method='POST')
            if info['status'] != 202:
                self.module.fail_json(msg='Failed to publish the repo', status_code=info['status'], response=info['msg'], url=url, distributor=publish_distributor)
        if self.wait_for_completion:
            self.verify_tasks_completed(json.load(response))
        return True

    def sync_repo(self, repo_id):
        url = '%s/pulp/api/v2/repositories/%s/actions/sync/' % (self.host, repo_id)
        response, info = fetch_url(self.module, url, data='', method='POST')
        if info['status'] != 202:
            self.module.fail_json(msg='Failed to schedule a sync of the repo.', status_code=info['status'], response=info['msg'], url=url)
        if self.wait_for_completion:
            self.verify_tasks_completed(json.load(response))
        return True

    def update_repo_distributor_config(self, repo_id, **kwargs):
        url = '%s/pulp/api/v2/repositories/%s/distributors/' % (self.host, repo_id)
        repo_config = self.get_repo_config_by_id(repo_id)
        for distributor in repo_config['distributors']:
            distributor_url = '%s%s/' % (url, distributor['id'])
            data = dict()
            data['distributor_config'] = dict()
            for key, value in kwargs.items():
                data['distributor_config'][key] = value
            response, info = fetch_url(self.module, distributor_url, data=json.dumps(data), method='PUT')
            if info['status'] != 202:
                self.module.fail_json(msg='Failed to set the relative url for the repository.', status_code=info['status'], response=info['msg'], url=url)

    def update_repo_importer_config(self, repo_id, **kwargs):
        url = '%s/pulp/api/v2/repositories/%s/importers/' % (self.host, repo_id)
        data = dict()
        importer_config = dict()
        for key, value in kwargs.items():
            if value is not None:
                importer_config[key] = value
        data['importer_config'] = importer_config
        if self.repo_type == 'rpm':
            data['importer_type_id'] = 'yum_importer'
        response, info = fetch_url(self.module, url, data=json.dumps(data), method='POST')
        if info['status'] != 202:
            self.module.fail_json(msg='Failed to set the repo importer configuration', status_code=info['status'], response=info['msg'], importer_config=importer_config, url=url)

    def set_repo_list(self):
        url = '%s/pulp/api/v2/repositories/?details=true' % self.host
        response, info = fetch_url(self.module, url, method='GET')
        if info['status'] != 200:
            self.module.fail_json(msg='Request failed', status_code=info['status'], response=info['msg'], url=url)
        self.repo_list = json.load(response)

    def verify_tasks_completed(self, response_dict):
        for task in response_dict['spawned_tasks']:
            task_url = '%s%s' % (self.host, task['_href'])
            while True:
                response, info = fetch_url(self.module, task_url, data='', method='GET')
                if info['status'] != 200:
                    self.module.fail_json(msg='Failed to check async task status.', status_code=info['status'], response=info['msg'], url=task_url)
                task_dict = json.load(response)
                if task_dict['state'] == 'finished':
                    return True
                if task_dict['state'] == 'error':
                    self.module.fail_json(msg='Asynchronous task failed to complete.', error=task_dict['error'])
                sleep(2)