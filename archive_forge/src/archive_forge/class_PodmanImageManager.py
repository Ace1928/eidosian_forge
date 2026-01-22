from __future__ import absolute_import, division, print_function
import json
import re
import shlex
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.containers.podman.plugins.module_utils.podman.common import run_podman_command
class PodmanImageManager(object):

    def __init__(self, module, results):
        super(PodmanImageManager, self).__init__()
        self.module = module
        self.results = results
        self.name = self.module.params.get('name')
        self.executable = self.module.get_bin_path(module.params.get('executable'), required=True)
        self.tag = self.module.params.get('tag')
        self.pull = self.module.params.get('pull')
        self.push = self.module.params.get('push')
        self.path = self.module.params.get('path')
        self.force = self.module.params.get('force')
        self.state = self.module.params.get('state')
        self.validate_certs = self.module.params.get('validate_certs')
        self.auth_file = self.module.params.get('auth_file')
        self.username = self.module.params.get('username')
        self.password = self.module.params.get('password')
        self.ca_cert_dir = self.module.params.get('ca_cert_dir')
        self.build = self.module.params.get('build')
        self.push_args = self.module.params.get('push_args')
        self.arch = self.module.params.get('arch')
        repo, repo_tag = parse_repository_tag(self.name)
        if repo_tag:
            self.name = repo
            self.tag = repo_tag
        delimiter = ':' if 'sha256' not in self.tag else '@'
        self.image_name = '{name}{d}{tag}'.format(name=self.name, d=delimiter, tag=self.tag)
        if self.state in ['present', 'build']:
            self.present()
        if self.state in ['absent']:
            self.absent()

    def _run(self, args, expected_rc=0, ignore_errors=False):
        cmd = ' '.join([self.executable] + [to_native(i) for i in args])
        self.module.log('PODMAN-IMAGE-DEBUG: %s' % cmd)
        self.results['podman_actions'].append(cmd)
        return run_podman_command(module=self.module, executable=self.executable, args=args, expected_rc=expected_rc, ignore_errors=ignore_errors)

    def _get_id_from_output(self, lines, startswith=None, contains=None, split_on=' ', maxsplit=1):
        layer_ids = []
        for line in lines.splitlines():
            if startswith and line.startswith(startswith) or (contains and contains in line):
                splitline = line.rsplit(split_on, maxsplit)
                layer_ids.append(splitline[1])
        if not layer_ids:
            layer_ids = lines.splitlines()
        return layer_ids[-1]

    def present(self):
        image = self.find_image()
        if image:
            digest_before = image[0].get('Digest', image[0].get('digest'))
        else:
            digest_before = None
        if not image or self.force:
            if self.path:
                self.results['actions'].append('Built image {image_name} from {path}'.format(image_name=self.image_name, path=self.path))
                if not self.module.check_mode:
                    self.results['image'], self.results['stdout'] = self.build_image()
                    image = self.results['image']
            else:
                self.results['actions'].append('Pulled image {image_name}'.format(image_name=self.image_name))
                if not self.module.check_mode:
                    image = self.results['image'] = self.pull_image()
            if not image:
                image = self.find_image()
            if not self.module.check_mode:
                digest_after = image[0].get('Digest', image[0].get('digest'))
                self.results['changed'] = digest_before != digest_after
            else:
                self.results['changed'] = True
        if self.push:
            if '/' in self.image_name:
                push_format_string = 'Pushed image {image_name}'
            else:
                push_format_string = 'Pushed image {image_name} to {dest}'
            self.results['actions'].append(push_format_string.format(image_name=self.image_name, dest=self.push_args['dest']))
            self.results['changed'] = True
            if not self.module.check_mode:
                self.results['image'], output = self.push_image()
                self.results['stdout'] += '\n' + output
        if image and (not self.results.get('image')):
            self.results['image'] = image

    def absent(self):
        image = self.find_image()
        image_id = self.find_image_id()
        if image:
            self.results['actions'].append('Removed image {name}'.format(name=self.name))
            self.results['changed'] = True
            self.results['image']['state'] = 'Deleted'
            if not self.module.check_mode:
                self.remove_image()
        elif image_id:
            self.results['actions'].append('Removed image with id {id}'.format(id=self.image_name))
            self.results['changed'] = True
            self.results['image']['state'] = 'Deleted'
            if not self.module.check_mode:
                self.remove_image_id()

    def find_image(self, image_name=None):
        if image_name is None:
            image_name = self.image_name
        args = ['image', 'ls', image_name, '--format', 'json']
        rc, images, err = self._run(args, ignore_errors=True)
        try:
            images = json.loads(images)
        except json.decoder.JSONDecodeError:
            self.module.fail_json(msg='Failed to parse JSON output from podman image ls: {out}'.format(out=images))
        if len(images) == 0:
            rc, out, err = self._run(['image', 'exists', image_name], ignore_errors=True)
            if rc == 0:
                inspect_json = self.inspect_image(image_name)
            else:
                return None
        if len(images) > 0:
            inspect_json = self.inspect_image(image_name)
        if self._is_target_arch(inspect_json, self.arch) or not self.arch:
            return images or inspect_json
        return None

    def _is_target_arch(self, inspect_json=None, arch=None):
        return arch and inspect_json[0]['Architecture'] == arch

    def find_image_id(self, image_id=None):
        if image_id is None:
            image_id = re.sub(':.*$', '', self.image_name)
        args = ['image', 'ls', '--quiet', '--no-trunc']
        rc, candidates, err = self._run(args, ignore_errors=True)
        candidates = [re.sub('^sha256:', '', c) for c in str.splitlines(candidates)]
        for c in candidates:
            if c.startswith(image_id):
                return image_id
        return None

    def inspect_image(self, image_name=None):
        if image_name is None:
            image_name = self.image_name
        args = ['inspect', image_name, '--format', 'json']
        rc, image_data, err = self._run(args)
        try:
            image_data = json.loads(image_data)
        except json.decoder.JSONDecodeError:
            self.module.fail_json(msg='Failed to parse JSON output from podman inspect: {out}'.format(out=image_data))
        if len(image_data) > 0:
            return image_data
        else:
            return None

    def pull_image(self, image_name=None):
        if image_name is None:
            image_name = self.image_name
        args = ['pull', image_name, '-q']
        if self.arch:
            args.extend(['--arch', self.arch])
        if self.auth_file:
            args.extend(['--authfile', self.auth_file])
        if self.username and self.password:
            cred_string = '{user}:{password}'.format(user=self.username, password=self.password)
            args.extend(['--creds', cred_string])
        if self.validate_certs is not None:
            if self.validate_certs:
                args.append('--tls-verify')
            else:
                args.append('--tls-verify=false')
        if self.ca_cert_dir:
            args.extend(['--cert-dir', self.ca_cert_dir])
        rc, out, err = self._run(args, ignore_errors=True)
        if rc != 0:
            if not self.pull:
                self.module.fail_json(msg='Failed to find image {image_name} locally, image pull set to {pull_bool}'.format(pull_bool=self.pull, image_name=image_name))
            else:
                self.module.fail_json(msg='Failed to pull image {image_name}'.format(image_name=image_name))
        return self.inspect_image(out.strip())

    def build_image(self):
        args = ['build']
        args.extend(['-t', self.image_name])
        if self.validate_certs is not None:
            if self.validate_certs:
                args.append('--tls-verify')
            else:
                args.append('--tls-verify=false')
        annotation = self.build.get('annotation')
        if annotation:
            for k, v in annotation.items():
                args.extend(['--annotation', '{k}={v}'.format(k=k, v=v)])
        if self.ca_cert_dir:
            args.extend(['--cert-dir', self.ca_cert_dir])
        if self.build.get('force_rm'):
            args.append('--force-rm')
        image_format = self.build.get('format')
        if image_format:
            args.extend(['--format', image_format])
        if not self.build.get('cache'):
            args.append('--no-cache')
        if self.build.get('rm'):
            args.append('--rm')
        containerfile = self.build.get('file')
        if containerfile:
            args.extend(['--file', containerfile])
        volume = self.build.get('volume')
        if volume:
            for v in volume:
                args.extend(['--volume', v])
        if self.auth_file:
            args.extend(['--authfile', self.auth_file])
        if self.username and self.password:
            cred_string = '{user}:{password}'.format(user=self.username, password=self.password)
            args.extend(['--creds', cred_string])
        extra_args = self.build.get('extra_args')
        if extra_args:
            args.extend(shlex.split(extra_args))
        target = self.build.get('target')
        if target:
            args.extend(['--target', target])
        args.append(self.path)
        rc, out, err = self._run(args, ignore_errors=True)
        if rc != 0:
            self.module.fail_json(msg='Failed to build image {image}: {out} {err}'.format(image=self.image_name, out=out, err=err))
        last_id = self._get_id_from_output(out, startswith='-->')
        return (self.inspect_image(last_id), out + err)

    def push_image(self):
        args = ['push']
        if self.validate_certs is not None:
            if self.validate_certs:
                args.append('--tls-verify')
            else:
                args.append('--tls-verify=false')
        if self.ca_cert_dir:
            args.extend(['--cert-dir', self.ca_cert_dir])
        if self.username and self.password:
            cred_string = '{user}:{password}'.format(user=self.username, password=self.password)
            args.extend(['--creds', cred_string])
        if self.auth_file:
            args.extend(['--authfile', self.auth_file])
        if self.push_args.get('compress'):
            args.append('--compress')
        push_format = self.push_args.get('format')
        if push_format:
            args.extend(['--format', push_format])
        if self.push_args.get('remove_signatures'):
            args.append('--remove-signatures')
        sign_by_key = self.push_args.get('sign_by')
        if sign_by_key:
            args.extend(['--sign-by', sign_by_key])
        args.append(self.image_name)
        dest = self.push_args.get('dest')
        dest_format_string = '{dest}/{image_name}'
        regexp = re.compile('/{name}(:{tag})?'.format(name=self.name, tag=self.tag))
        if not dest:
            if '/' not in self.name:
                self.module.fail_json(msg="'push_args['dest']' is required when pushing images that do not have the remote registry in the image name")
        elif regexp.search(dest):
            dest = regexp.sub('', dest)
            self.module.warn("Image name and tag are automatically added to push_args['dest']. Destination changed to {dest}".format(dest=dest))
        if dest and dest.endswith('/'):
            dest = dest[:-1]
        transport = self.push_args.get('transport')
        if transport:
            if not dest:
                self.module.fail_json("'push_args['transport'] requires 'push_args['dest'] but it was not provided.")
            if transport == 'docker':
                dest_format_string = '{transport}://{dest}'
            elif transport == 'ostree':
                dest_format_string = '{transport}:{name}@{dest}'
            else:
                dest_format_string = '{transport}:{dest}'
        dest_string = dest_format_string.format(transport=transport, name=self.name, dest=dest, image_name=self.image_name)
        if '/' not in self.name:
            args.append(dest_string)
        rc, out, err = self._run(args, ignore_errors=True)
        if rc != 0:
            self.module.fail_json(msg='Failed to push image {image_name}: {err}'.format(image_name=self.image_name, err=err))
        last_id = self._get_id_from_output(out + err, contains=':', split_on=':')
        return (self.inspect_image(last_id), out + err)

    def remove_image(self, image_name=None):
        if image_name is None:
            image_name = self.image_name
        args = ['rmi', image_name]
        if self.force:
            args.append('--force')
        rc, out, err = self._run(args, ignore_errors=True)
        if rc != 0:
            self.module.fail_json(msg='Failed to remove image {image_name}. {err}'.format(image_name=image_name, err=err))
        return out

    def remove_image_id(self, image_id=None):
        if image_id is None:
            image_id = re.sub(':.*$', '', self.image_name)
        args = ['rmi', image_id]
        if self.force:
            args.append('--force')
        rc, out, err = self._run(args, ignore_errors=True)
        if rc != 0:
            self.module.fail_json(msg='Failed to remove image with id {image_id}. {err}'.format(image_id=image_id, err=err))
        return out