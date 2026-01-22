from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
def _AddServerlessRoutingInfo(parser, support_serverless_deployment=False):
    """Adds serverless routing info arguments for network endpoint groups."""
    serverless_group_help = '      The serverless routing configurations are only valid when endpoint type\n      of the network endpoint group is `serverless`.\n  '
    serverless_group = parser.add_group(help=serverless_group_help, mutex=True)
    cloud_run_group_help = '      Configuration for a Cloud Run network endpoint group. Cloud Run service\n      must be provided explicitly or in the URL mask. Cloud Run tag is optional,\n      and may be provided explicitly or in the URL mask.\n  '
    cloud_run_group = serverless_group.add_group(help=cloud_run_group_help)
    cloud_run_service_help = '      Cloud Run service name to add to the Serverless network endpoint groups\n      (NEG). The service must be in the same project and the same region as the\n      Serverless NEG.\n  '
    cloud_run_group.add_argument('--cloud-run-service', help=cloud_run_service_help)
    cloud_run_tag_help = '      Cloud Run tag represents the "named revision" to provide additional\n      fine-grained traffic routing configuration.\n  '
    cloud_run_group.add_argument('--cloud-run-tag', help=cloud_run_tag_help)
    cloud_run_url_mask_help = '      A template to parse service and tag fields from a request URL. URL mask\n      allows for routing to multiple Run services without having to create\n      multiple network endpoint groups and backend services.\n  '
    cloud_run_group.add_argument('--cloud-run-url-mask', help=cloud_run_url_mask_help)
    app_engine_group_help = '      Configuration for an App Engine network endpoint group. Both App Engine\n      service and version are optional, and may be provided explicitly or in the\n      URL mask. The `app-engine-app` flag is only used for default routing. The\n      App Engine app must be in the same project as the Serverless network\n      endpoint groups (NEG).\n  '
    app_engine_group = serverless_group.add_group(help=app_engine_group_help)
    app_engine_group.add_argument('--app-engine-app', action=arg_parsers.StoreTrueFalseAction, help='If set, the default routing is used.')
    app_engine_group.add_argument('--app-engine-service', help='Optional serving service to add to the Serverless NEG.')
    app_engine_group.add_argument('--app-engine-version', help='Optional serving version to add to the Serverless NEG.')
    app_engine_url_mask_help = '      A template to parse service and version fields from a request URL. URL\n      mask allows for routing to multiple App Engine services without having\n      to create multiple network endpoint groups and backend services.\n  '
    app_engine_group.add_argument('--app-engine-url-mask', help=app_engine_url_mask_help)
    cloud_function_group_help = '      Configuration for a Cloud Function network endpoint group. Cloud Function\n      name must be provided explicitly or in the URL mask.\n  '
    cloud_function_group = serverless_group.add_group(help=cloud_function_group_help)
    cloud_function_name_help = '      Cloud Function name to add to the Serverless NEG. The function must be in\n      the same project and the same region as the Serverless network endpoint\n      groups (NEG).\n  '
    cloud_function_group.add_argument('--cloud-function-name', help=cloud_function_name_help)
    cloud_function_url_mask_help = '      A template to parse function field from a request URL. URL mask allows\n      for routing to multiple Cloud Functions without having to create multiple\n      network endpoint groups and backend services.\n  '
    cloud_function_group.add_argument('--cloud-function-url-mask', help=cloud_function_url_mask_help)
    if support_serverless_deployment:
        serverless_deployment_group_help = '        Configuration for a Serverless network endpoint group.\n        Serverless NEGs support all serverless backends and are the only way to\n        setup a network endpoint group for Cloud API Gateways.\n\n        To create a serverless NEG with a Cloud Run, Cloud Functions or App\n        Engine endpoint, you can either use the previously-listed Cloud Run,\n        Cloud Functions or App Engine-specific properties, OR, you can use the\n        following generic properties that are compatible with all serverless\n        platforms, including API Gateway: serverless-deployment-platform,\n        serverless-deployment-resource, serverless-deployment-url-mask, and\n        serverless-deployment-version.\n    '
        serverless_deployment_group = serverless_group.add_group(help=serverless_deployment_group_help)
        serverless_deployment_platform_help = '        The platform of the NEG backend target(s). Possible values:\n\n          * API Gateway: apigateway.googleapis.com\n          * App Engine: appengine.googleapis.com\n          * Cloud Functions: cloudfunctions.googleapis.com\n          * Cloud Run: run.googleapis.com\n    '
        serverless_deployment_group.add_argument('--serverless-deployment-platform', help=serverless_deployment_platform_help)
        serverless_deployment_resource_help = '        The user-defined name of the workload/instance. This value must be\n        provided explicitly or using the --serverless-deployment-url-mask\n        option. The resource identified by this value is platform-specific and\n        is as follows:\n\n          * API Gateway: The gateway ID\n          * App Engine: The service name\n          * Cloud Functions: The function name\n          * Cloud Run: The service name\n    '
        serverless_deployment_group.add_argument('--serverless-deployment-resource', help=serverless_deployment_resource_help)
        serverless_deployment_version_help = '        The optional resource version. The version identified by this value is\n        platform-specific and is as follows:\n\n          * API Gateway: Unused\n          * App Engine: The service version\n          * Cloud Functions: Unused\n          * Cloud Run: The service tag\n    '
        serverless_deployment_group.add_argument('--serverless-deployment-version', help=serverless_deployment_version_help)
        serverless_deployment_url_mask_help = "        A template to parse platform-specific fields from a request URL. URL\n        mask allows for routing to multiple resources on the same serverless\n        platform without having to create multiple network endpoint groups and\n        backend resources. The fields parsed by this template are\n        platform-specific and are as follows:\n\n          * API Gateway: The 'gateway' ID\n          * App Engine: The 'service' and 'version'\n          * Cloud Functions: The 'function' name\n          * Cloud Run: The 'service' and 'tag'\n    "
        serverless_deployment_group.add_argument('--serverless-deployment-url-mask', help=serverless_deployment_url_mask_help)