from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.bigtable import clusters
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.bigtable import arguments
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
@base.ReleaseTracks(base.ReleaseTrack.GA, base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class CreateInstance(base.CreateCommand):
    """Create a new Bigtable instance."""
    detailed_help = {'EXAMPLES': textwrap.dedent('          To create an instance with id `my-instance-id` with a cluster located\n          in `us-east1-c`, run:\n\n            $ {command} my-instance-id --display-name="My Instance" --cluster-config=id=my-cluster-id,zone=us-east1-c\n\n          To create an instance with multiple clusters, run:\n\n            $ {command} my-instance-id --display-name="My Instance" --cluster-config=id=my-cluster-id-1,zone=us-east1-c --cluster-config=id=my-cluster-id-2,zone=us-west1-c,nodes=3\n\n          To create an instance with `HDD` storage and `10` nodes, run:\n\n            $ {command} my-hdd-instance --display-name="HDD Instance" --cluster-storage-type=HDD --cluster-config=id=my-cluster-id,zone=us-east1-c,nodes=10\n\n          ')}

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        arguments.ArgAdder(parser).AddInstanceDisplayName(required=True).AddClusterConfig().AddDeprecatedCluster().AddDeprecatedClusterZone().AddDeprecatedClusterNodes().AddClusterStorage().AddAsync().AddDeprecatedInstanceType()
        arguments.AddInstanceResourceArg(parser, 'to create', positional=True)
        parser.display_info.AddCacheUpdater(arguments.InstanceCompleter)

    def Run(self, args):
        """Executes the instances create command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Some value that we want to have printed later.
    """
        return self._Run(args)

    def _Run(self, args):
        """Implements Run() with different possible features flags."""
        cli = util.GetAdminClient()
        ref = args.CONCEPTS.instance.Parse()
        parent_ref = resources.REGISTRY.Create('bigtableadmin.projects', projectId=ref.projectsId)
        msgs = util.GetAdminMessages()
        instance_type = msgs.Instance.TypeValueValuesEnum(args.instance_type)
        new_clusters = self._Clusters(args)
        clusters_properties = []
        for cluster_id, cluster in sorted(new_clusters.items()):
            clusters_properties.append(msgs.CreateInstanceRequest.ClustersValue.AdditionalProperty(key=cluster_id, value=cluster))
        msg = msgs.CreateInstanceRequest(instanceId=ref.Name(), parent=parent_ref.RelativeName(), instance=msgs.Instance(displayName=args.display_name, type=instance_type), clusters=msgs.CreateInstanceRequest.ClustersValue(additionalProperties=clusters_properties))
        result = cli.projects_instances.Create(msg)
        operation_ref = util.GetOperationRef(result)
        if args.async_:
            log.CreatedResource(operation_ref.RelativeName(), kind='bigtable instance {0}'.format(ref.Name()), is_async=True)
            return result
        return util.AwaitInstance(operation_ref, 'Creating bigtable instance {0}'.format(ref.Name()))

    def _Clusters(self, args):
        """Get the clusters configs from command arguments.

    Args:
      args: the argparse namespace from Run().

    Returns:
      A dict mapping from cluster id to msg.Cluster.
    """
        msgs = util.GetAdminMessages()
        storage_type = msgs.Cluster.DefaultStorageTypeValueValuesEnum(args.cluster_storage_type.upper())
        if args.cluster_config is not None:
            if args.cluster is not None or args.cluster_zone is not None or args.cluster_num_nodes is not None:
                raise exceptions.InvalidArgumentException('--cluster-config --cluster --cluster-zone --cluster-num-nodes', 'Use --cluster-config or the combination of --cluster, --cluster-zone and --cluster-num-nodes to specify cluster(s), not both.')
            self._ValidateClusterConfigArgs(args.cluster_config)
            new_clusters = {}
            for cluster_dict in args.cluster_config:
                nodes = cluster_dict.get('nodes', 1)
                cluster = msgs.Cluster(serveNodes=nodes, defaultStorageType=storage_type, location=util.LocationUrl(cluster_dict['zone']))
                if 'kms-key' in cluster_dict:
                    cluster.encryptionConfig = msgs.EncryptionConfig(kmsKeyName=cluster_dict['kms-key'])
                if 'autoscaling-min-nodes' in cluster_dict or 'autoscaling-max-nodes' in cluster_dict or 'autoscaling-cpu-target' in cluster_dict:
                    if 'autoscaling-storage-target' in cluster_dict:
                        storage_target = cluster_dict['autoscaling-storage-target']
                    else:
                        storage_target = None
                    cluster.clusterConfig = clusters.BuildClusterConfig(autoscaling_min=cluster_dict['autoscaling-min-nodes'], autoscaling_max=cluster_dict['autoscaling-max-nodes'], autoscaling_cpu_target=cluster_dict['autoscaling-cpu-target'], autoscaling_storage_target=storage_target)
                    cluster.serveNodes = None
                new_clusters[cluster_dict['id']] = cluster
            return new_clusters
        elif args.cluster is not None:
            if args.cluster_zone is None:
                raise exceptions.InvalidArgumentException('--cluster-zone', '--cluster-zone must be specified.')
            cluster = msgs.Cluster(serveNodes=arguments.ProcessInstanceTypeAndNodes(args), defaultStorageType=storage_type, location=util.LocationUrl(args.cluster_zone))
            return {args.cluster: cluster}
        else:
            raise exceptions.InvalidArgumentException('--cluster --cluster-config', 'Use --cluster-config to specify cluster(s).')

    def _ValidateClusterConfigArgs(self, cluster_config):
        """Validates arguments in cluster-config as a repeated dict."""
        for cluster_dict in cluster_config:
            if 'autoscaling-min-nodes' in cluster_dict or 'autoscaling-max-nodes' in cluster_dict or 'autoscaling-cpu-target' in cluster_dict or ('autoscaling-storage-target' in cluster_dict):
                if 'nodes' in cluster_dict:
                    raise exceptions.InvalidArgumentException('--autoscaling-min-nodes --autoscaling-max-nodes --autoscaling-cpu-target --autoscaling-storage-target', 'At most one of nodes | autoscaling-min-nodes autoscaling-max-nodes autoscaling-cpu-target autoscaling-storage-target may be specified in --cluster-config')
                if 'autoscaling-min-nodes' not in cluster_dict or 'autoscaling-max-nodes' not in cluster_dict or 'autoscaling-cpu-target' not in cluster_dict:
                    raise exceptions.InvalidArgumentException('--autoscaling-min-nodes --autoscaling-max-nodes --autoscaling-cpu-target', 'All of --autoscaling-min-nodes --autoscaling-max-nodes --autoscaling-cpu-target must be set to enable Autoscaling.')