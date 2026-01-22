from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
class InstancesService(base_api.BaseApiService):
    """Service class for the instances resource."""
    _NAME = 'instances'

    def __init__(self, client):
        super(SqladminV1beta4.InstancesService, self).__init__(client)
        self._upload_configs = {}

    def AcquireSsrsLease(self, request, global_params=None):
        """Acquire a lease for the setup of SQL Server Reporting Services (SSRS).

      Args:
        request: (SqlInstancesAcquireSsrsLeaseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SqlInstancesAcquireSsrsLeaseResponse) The response message.
      """
        config = self.GetMethodConfig('AcquireSsrsLease')
        return self._RunMethod(config, request, global_params=global_params)
    AcquireSsrsLease.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.acquireSsrsLease', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/acquireSsrsLease', request_field='instancesAcquireSsrsLeaseRequest', request_type_name='SqlInstancesAcquireSsrsLeaseRequest', response_type_name='SqlInstancesAcquireSsrsLeaseResponse', supports_download=False)

    def AddServerCa(self, request, global_params=None):
        """Add a new trusted Certificate Authority (CA) version for the specified instance. Required to prepare for a certificate rotation. If a CA version was previously added but never used in a certificate rotation, this operation replaces that version. There cannot be more than one CA version waiting to be rotated in.

      Args:
        request: (SqlInstancesAddServerCaRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('AddServerCa')
        return self._RunMethod(config, request, global_params=global_params)
    AddServerCa.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.addServerCa', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/addServerCa', request_field='', request_type_name='SqlInstancesAddServerCaRequest', response_type_name='Operation', supports_download=False)

    def Clone(self, request, global_params=None):
        """Creates a Cloud SQL instance as a clone of the source instance. Using this operation might cause your instance to restart.

      Args:
        request: (SqlInstancesCloneRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Clone')
        return self._RunMethod(config, request, global_params=global_params)
    Clone.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.clone', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/clone', request_field='instancesCloneRequest', request_type_name='SqlInstancesCloneRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Cloud SQL instance.

      Args:
        request: (SqlInstancesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='sql.instances.delete', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=['finalBackupDescription', 'finalBackupExpiryTime', 'finalBackupTtlDays', 'skipFinalBackup'], relative_path='sql/v1beta4/projects/{project}/instances/{instance}', request_field='', request_type_name='SqlInstancesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Demote(self, request, global_params=None):
        """Demotes an existing standalone instance to be a Cloud SQL read replica for an external database server.

      Args:
        request: (SqlInstancesDemoteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Demote')
        return self._RunMethod(config, request, global_params=global_params)
    Demote.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.demote', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/demote', request_field='instancesDemoteRequest', request_type_name='SqlInstancesDemoteRequest', response_type_name='Operation', supports_download=False)

    def DemoteMaster(self, request, global_params=None):
        """Demotes the stand-alone instance to be a Cloud SQL read replica for an external database server.

      Args:
        request: (SqlInstancesDemoteMasterRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('DemoteMaster')
        return self._RunMethod(config, request, global_params=global_params)
    DemoteMaster.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.demoteMaster', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/demoteMaster', request_field='instancesDemoteMasterRequest', request_type_name='SqlInstancesDemoteMasterRequest', response_type_name='Operation', supports_download=False)

    def Export(self, request, global_params=None):
        """Exports data from a Cloud SQL instance to a Cloud Storage bucket as a SQL dump or CSV file.

      Args:
        request: (SqlInstancesExportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Export')
        return self._RunMethod(config, request, global_params=global_params)
    Export.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.export', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/export', request_field='instancesExportRequest', request_type_name='SqlInstancesExportRequest', response_type_name='Operation', supports_download=False)

    def Failover(self, request, global_params=None):
        """Initiates a manual failover of a high availability (HA) primary instance to a standby instance, which becomes the primary instance. Users are then rerouted to the new primary. For more information, see the [Overview of high availability](https://cloud.google.com/sql/docs/mysql/high-availability) page in the Cloud SQL documentation. If using Legacy HA (MySQL only), this causes the instance to failover to its failover replica instance.

      Args:
        request: (SqlInstancesFailoverRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Failover')
        return self._RunMethod(config, request, global_params=global_params)
    Failover.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.failover', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/failover', request_field='instancesFailoverRequest', request_type_name='SqlInstancesFailoverRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a resource containing information about a Cloud SQL instance.

      Args:
        request: (SqlInstancesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DatabaseInstance) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='sql.instances.get', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}', request_field='', request_type_name='SqlInstancesGetRequest', response_type_name='DatabaseInstance', supports_download=False)

    def Import(self, request, global_params=None):
        """Imports data into a Cloud SQL instance from a SQL dump or CSV file in Cloud Storage.

      Args:
        request: (SqlInstancesImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.import', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/import', request_field='instancesImportRequest', request_type_name='SqlInstancesImportRequest', response_type_name='Operation', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new Cloud SQL instance.

      Args:
        request: (SqlInstancesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.insert', ordered_params=['project'], path_params=['project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances', request_field='databaseInstance', request_type_name='SqlInstancesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists instances under a given project.

      Args:
        request: (SqlInstancesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstancesListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='sql.instances.list', ordered_params=['project'], path_params=['project'], query_params=['filter', 'maxResults', 'pageToken'], relative_path='sql/v1beta4/projects/{project}/instances', request_field='', request_type_name='SqlInstancesListRequest', response_type_name='InstancesListResponse', supports_download=False)

    def ListServerCas(self, request, global_params=None):
        """Lists all of the trusted Certificate Authorities (CAs) for the specified instance. There can be up to three CAs listed: the CA that was used to sign the certificate that is currently in use, a CA that has been added but not yet used to sign a certificate, and a CA used to sign a certificate that has previously rotated out.

      Args:
        request: (SqlInstancesListServerCasRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstancesListServerCasResponse) The response message.
      """
        config = self.GetMethodConfig('ListServerCas')
        return self._RunMethod(config, request, global_params=global_params)
    ListServerCas.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='sql.instances.listServerCas', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/listServerCas', request_field='', request_type_name='SqlInstancesListServerCasRequest', response_type_name='InstancesListServerCasResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Partially updates settings of a Cloud SQL instance by merging the request with the current configuration. This method supports patch semantics.

      Args:
        request: (SqlInstancesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='sql.instances.patch', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}', request_field='databaseInstance', request_type_name='SqlInstancesPatchRequest', response_type_name='Operation', supports_download=False)

    def PromoteReplica(self, request, global_params=None):
        """Promotes the read replica instance to be a stand-alone Cloud SQL instance. Using this operation might cause your instance to restart.

      Args:
        request: (SqlInstancesPromoteReplicaRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('PromoteReplica')
        return self._RunMethod(config, request, global_params=global_params)
    PromoteReplica.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.promoteReplica', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=['failover'], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/promoteReplica', request_field='', request_type_name='SqlInstancesPromoteReplicaRequest', response_type_name='Operation', supports_download=False)

    def Reencrypt(self, request, global_params=None):
        """Reencrypt CMEK instance with latest key version.

      Args:
        request: (SqlInstancesReencryptRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Reencrypt')
        return self._RunMethod(config, request, global_params=global_params)
    Reencrypt.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.reencrypt', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/reencrypt', request_field='instancesReencryptRequest', request_type_name='SqlInstancesReencryptRequest', response_type_name='Operation', supports_download=False)

    def ReleaseSsrsLease(self, request, global_params=None):
        """Release a lease for the setup of SQL Server Reporting Services (SSRS).

      Args:
        request: (SqlInstancesReleaseSsrsLeaseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SqlInstancesReleaseSsrsLeaseResponse) The response message.
      """
        config = self.GetMethodConfig('ReleaseSsrsLease')
        return self._RunMethod(config, request, global_params=global_params)
    ReleaseSsrsLease.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.releaseSsrsLease', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/releaseSsrsLease', request_field='', request_type_name='SqlInstancesReleaseSsrsLeaseRequest', response_type_name='SqlInstancesReleaseSsrsLeaseResponse', supports_download=False)

    def ResetSslConfig(self, request, global_params=None):
        """Deletes all client certificates and generates a new server SSL certificate for the instance.

      Args:
        request: (SqlInstancesResetSslConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ResetSslConfig')
        return self._RunMethod(config, request, global_params=global_params)
    ResetSslConfig.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.resetSslConfig', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/resetSslConfig', request_field='', request_type_name='SqlInstancesResetSslConfigRequest', response_type_name='Operation', supports_download=False)

    def Restart(self, request, global_params=None):
        """Restarts a Cloud SQL instance.

      Args:
        request: (SqlInstancesRestartRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Restart')
        return self._RunMethod(config, request, global_params=global_params)
    Restart.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.restart', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/restart', request_field='', request_type_name='SqlInstancesRestartRequest', response_type_name='Operation', supports_download=False)

    def RestoreBackup(self, request, global_params=None):
        """Restores a backup of a Cloud SQL instance. Using this operation might cause your instance to restart.

      Args:
        request: (SqlInstancesRestoreBackupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RestoreBackup')
        return self._RunMethod(config, request, global_params=global_params)
    RestoreBackup.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.restoreBackup', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/restoreBackup', request_field='instancesRestoreBackupRequest', request_type_name='SqlInstancesRestoreBackupRequest', response_type_name='Operation', supports_download=False)

    def RotateServerCa(self, request, global_params=None):
        """Rotates the server certificate to one signed by the Certificate Authority (CA) version previously added with the addServerCA method.

      Args:
        request: (SqlInstancesRotateServerCaRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('RotateServerCa')
        return self._RunMethod(config, request, global_params=global_params)
    RotateServerCa.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.rotateServerCa', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/rotateServerCa', request_field='instancesRotateServerCaRequest', request_type_name='SqlInstancesRotateServerCaRequest', response_type_name='Operation', supports_download=False)

    def StartReplica(self, request, global_params=None):
        """Starts the replication in the read replica instance.

      Args:
        request: (SqlInstancesStartReplicaRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('StartReplica')
        return self._RunMethod(config, request, global_params=global_params)
    StartReplica.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.startReplica', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/startReplica', request_field='', request_type_name='SqlInstancesStartReplicaRequest', response_type_name='Operation', supports_download=False)

    def StopReplica(self, request, global_params=None):
        """Stops the replication in the read replica instance.

      Args:
        request: (SqlInstancesStopReplicaRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('StopReplica')
        return self._RunMethod(config, request, global_params=global_params)
    StopReplica.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.stopReplica', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/stopReplica', request_field='', request_type_name='SqlInstancesStopReplicaRequest', response_type_name='Operation', supports_download=False)

    def Switchover(self, request, global_params=None):
        """Switches over from the primary instance to a replica instance.

      Args:
        request: (SqlInstancesSwitchoverRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Switchover')
        return self._RunMethod(config, request, global_params=global_params)
    Switchover.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.switchover', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=['dbTimeout'], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/switchover', request_field='', request_type_name='SqlInstancesSwitchoverRequest', response_type_name='Operation', supports_download=False)

    def TruncateLog(self, request, global_params=None):
        """Truncate MySQL general and slow query log tables MySQL only.

      Args:
        request: (SqlInstancesTruncateLogRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('TruncateLog')
        return self._RunMethod(config, request, global_params=global_params)
    TruncateLog.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.instances.truncateLog', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/truncateLog', request_field='instancesTruncateLogRequest', request_type_name='SqlInstancesTruncateLogRequest', response_type_name='Operation', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates settings of a Cloud SQL instance. Using this operation might cause your instance to restart.

      Args:
        request: (SqlInstancesUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='sql.instances.update', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}', request_field='databaseInstance', request_type_name='SqlInstancesUpdateRequest', response_type_name='Operation', supports_download=False)