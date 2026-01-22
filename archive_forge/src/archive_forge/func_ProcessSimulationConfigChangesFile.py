from googlecloudsdk.command_lib.network_management.simulation import util
from googlecloudsdk.core import properties
def ProcessSimulationConfigChangesFile(unused_ref, args, request):
    """Reads the firewall-service, route-service exported resources configs and transform them into the API accepted format and update the request proto."""
    if args.proposed_config_file:
        api_version = util.GetSimulationApiVersionFromArgs(args)
        request.simulation = util.PrepareSimulationChanges(args.proposed_config_file, api_version, file_format=args.file_format, simulation_type=args.simulation_type, original_config_file=args.original_config_file)
    return request