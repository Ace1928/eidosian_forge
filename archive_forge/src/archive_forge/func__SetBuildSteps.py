from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import config
from googlecloudsdk.api_lib.cloudbuild import logs as cb_logs
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags
from googlecloudsdk.command_lib.builds import staging_bucket_util
from googlecloudsdk.command_lib.cloudbuild import execution
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def _SetBuildSteps(tag, no_cache, messages, substitutions, arg_config, no_source, source, timeout_str, buildpack, client_tag):
    """Set build steps."""
    if tag is not None:
        if properties.VALUES.builds.check_tag.GetBool() and (not any((reg in tag for reg in _SUPPORTED_REGISTRIES))):
            raise c_exceptions.InvalidArgumentException('--tag', 'Tag value must be in the *gcr.io* or *pkg.dev* namespace.')
        if properties.VALUES.builds.use_kaniko.GetBool():
            if no_cache:
                ttl = '0h'
            else:
                ttl = '{}h'.format(properties.VALUES.builds.kaniko_cache_ttl.Get())
            build_config = messages.Build(steps=[messages.BuildStep(name=properties.VALUES.builds.kaniko_image.Get(), args=['--destination', tag, '--cache', '--cache-ttl', ttl, '--cache-dir', ''])], timeout=timeout_str, substitutions=cloudbuild_util.EncodeSubstitutions(substitutions, messages))
        else:
            if no_cache:
                raise c_exceptions.InvalidArgumentException('no-cache', 'Cannot specify --no-cache if builds/use_kaniko property is False')
            if not no_source and os.path.isdir(source):
                found = False
                for filename in os.listdir(source):
                    if filename == 'Dockerfile':
                        found = True
                        break
                if not found:
                    raise c_exceptions.InvalidArgumentException('source', 'Dockerfile required when specifying --tag')
            build_config = messages.Build(images=[tag], steps=[messages.BuildStep(name='gcr.io/cloud-builders/docker', args=['build', '--network', 'cloudbuild', '--no-cache', '-t', tag, '.'])], timeout=timeout_str, substitutions=cloudbuild_util.EncodeSubstitutions(substitutions, messages))
    elif buildpack is not None:
        if not buildpack:
            raise c_exceptions.InvalidArgumentException('--pack', 'Image value must not be empty.')
        if buildpack[0].get('image') is None:
            raise c_exceptions.InvalidArgumentException('--pack', 'Image value must not be empty.')
        image = buildpack[0].get('image')
        if properties.VALUES.builds.check_tag.GetBool() and (not any((reg in image for reg in _SUPPORTED_REGISTRIES))):
            raise c_exceptions.InvalidArgumentException('--pack', 'Image value must be in the *gcr.io* or *pkg.dev* namespace')
        env = buildpack[0].get('env')
        envs = buildpack[0].get('envs')
        builder = buildpack[0].get('builder')
        steps = []
        pack_args = ['build', image, '--network', 'cloudbuild']
        build_tags = [_GetBuildTag(builder)]
        if env is not None:
            pack_args.append('--env')
            pack_args.append(env)
        if envs is not None:
            for env in envs:
                pack_args.append('--env')
                pack_args.append(env)
        if builder is not None:
            pack_args.append('--builder')
            pack_args.append(builder)
        else:
            default_buildpacks_builder = 'gcr.io/buildpacks/builder:latest'
            build_tags.append(_GetBuildTag(default_buildpacks_builder))
            steps = [messages.BuildStep(name='gcr.io/k8s-skaffold/pack', entrypoint='pack', args=['config', 'default-builder', default_buildpacks_builder])]
        steps.append(messages.BuildStep(name='gcr.io/k8s-skaffold/pack', entrypoint='pack', args=pack_args))
        client_tag = 'other' if client_tag is None else client_tag
        cloudbuild_tags = list(map(lambda x: 'gcp-runtimes-builder-' + x + '-' + client_tag, build_tags))
        build_config = messages.Build(images=[image], steps=steps, timeout=timeout_str, substitutions=cloudbuild_util.EncodeSubstitutions(substitutions, messages), tags=cloudbuild_tags)
    else:
        if no_cache:
            raise c_exceptions.ConflictingArgumentsException('--config', '--no-cache')
        if not arg_config:
            raise c_exceptions.InvalidArgumentException('--config', 'Config file path must not be empty.')
        build_config = config.LoadCloudbuildConfigFromPath(arg_config, messages, params=substitutions)
    if timeout_str:
        build_config.timeout = timeout_str
    return build_config