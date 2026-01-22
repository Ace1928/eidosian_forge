from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.vmware import util
from googlecloudsdk.command_lib.util.apis import arg_utils
class LoggingServersClient(util.VmwareClientBase):
    """Cloud VMware LoggingServers client."""

    def __init__(self):
        super(LoggingServersClient, self).__init__()
        self.service = self.client.projects_locations_privateClouds_loggingServers

    def Get(self, resource):
        request = self.messages.VmwareengineProjectsLocationsPrivateCloudsLoggingServersGetRequest(name=resource.RelativeName())
        return self.service.Get(request)

    def Create(self, resource, hostname, source_type, protocol, port):
        parent = resource.Parent().RelativeName()
        logging_server = self.messages.LoggingServer(hostname=hostname)
        logging_server.sourceType = self.GetSourceType(source_type)
        logging_server.protocol = self.GetProtocol(protocol)
        logging_server.port = port
        request = self.messages.VmwareengineProjectsLocationsPrivateCloudsLoggingServersCreateRequest(parent=parent, loggingServer=logging_server, loggingServerId=resource.Name())
        return self.service.Create(request)

    def Delete(self, resource):
        request = self.messages.VmwareengineProjectsLocationsPrivateCloudsLoggingServersDeleteRequest(name=resource.RelativeName())
        return self.service.Delete(request)

    def List(self, private_cloud_resource):
        private_cloud = private_cloud_resource.RelativeName()
        request = self.messages.VmwareengineProjectsLocationsPrivateCloudsLoggingServersListRequest(parent=private_cloud)
        return list_pager.YieldFromList(self.service, request, batch_size_attribute='pageSize', field='loggingServers')

    def Update(self, resource, hostname=None, source_type=None, protocol=None, port=None):
        logging_server = self.Get(resource)
        update_mask = []
        if hostname is not None:
            logging_server.hostname = hostname
            update_mask.append('hostname')
        if source_type is not None:
            logging_server.sourceType = self.GetSourceType(source_type)
            update_mask.append('source_type')
        if protocol is not None:
            logging_server.protocol = self.GetProtocol(protocol)
            update_mask.append('protocol')
        if port is not None:
            logging_server.port = port
            update_mask.append('port')
        request = self.messages.VmwareengineProjectsLocationsPrivateCloudsLoggingServersPatchRequest(loggingServer=logging_server, name=resource.RelativeName(), updateMask=','.join(update_mask))
        return self.service.Patch(request)

    def GetSourceType(self, source_type):
        source_type_enum = arg_utils.ChoiceEnumMapper(arg_name='source_type', message_enum=self.messages.LoggingServer.SourceTypeValueValuesEnum, include_filter=lambda x: 'SOURCE_TYPE_UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(source_type))
        return source_type_enum

    def GetProtocol(self, protocol):
        protocol_enum = arg_utils.ChoiceEnumMapper(arg_name='protocol', message_enum=self.messages.LoggingServer.ProtocolValueValuesEnum, include_filter=lambda x: 'PROTOCOL_UNSPECIFIED' not in x).GetEnumForChoice(arg_utils.EnumNameToChoice(protocol))
        return protocol_enum