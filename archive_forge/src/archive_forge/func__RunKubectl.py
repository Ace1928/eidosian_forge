from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import time
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import image_versions_util as image_versions_command_util
from googlecloudsdk.command_lib.composer import resource_args
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
def _RunKubectl(self, args, env_obj):
    cluster_id = env_obj.config.gkeCluster
    cluster_location_id = command_util.ExtractGkeClusterLocationId(env_obj)
    tty = 'no-tty' not in args
    with command_util.TemporaryKubeconfig(cluster_location_id, cluster_id):
        try:
            image_version = env_obj.config.softwareConfig.imageVersion
            kubectl_ns = command_util.FetchKubectlNamespace(image_version)
            pod = command_util.GetGkePod(pod_substr=WORKER_POD_SUBSTR, kubectl_namespace=kubectl_ns)
            log.status.Print('Executing within the following Kubernetes cluster namespace: {}'.format(kubectl_ns))
            kubectl_args = ['exec', pod, '--stdin']
            if tty:
                kubectl_args.append('--tty')
            kubectl_args.extend(['--container', WORKER_CONTAINER, '--'])
            if args.tree:
                kubectl_args.extend(['python', '-m', 'pipdeptree', '--warn'])
            else:
                kubectl_args.extend(['python', '-m', 'pip', 'list'])
            command_util.RunKubectlCommand(command_util.AddKubectlNamespace(kubectl_ns, kubectl_args), out_func=log.out.Print)
        except command_util.KubectlError as e:
            raise self.ConvertKubectlError(e, env_obj)