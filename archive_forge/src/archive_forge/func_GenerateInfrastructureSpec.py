from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.command_lib.iam import iam_util
def GenerateInfrastructureSpec(args):
    """Generate InfrastructureSpec From Arguments."""
    module = dataplex_api.GetMessageModule()
    compute_resource = module.GoogleCloudDataplexV1EnvironmentInfrastructureSpecComputeResources(diskSizeGb=args.compute_disk_size_gb, nodeCount=args.compute_node_count, maxNodeCount=args.compute_max_node_count)
    os_image_runtime = module.GoogleCloudDataplexV1EnvironmentInfrastructureSpecOsImageRuntime(imageVersion=args.os_image_version, javaLibraries=args.os_image_java_libraries, pythonPackages=args.os_image_python_packages, properties=args.os_image_properties)
    infrastructure_spec = module.GoogleCloudDataplexV1EnvironmentInfrastructureSpec(compute=compute_resource, osImage=os_image_runtime)
    return infrastructure_spec