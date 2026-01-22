from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from typing import Generator
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.fleet import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import resources
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha import gkehub_v1alpha_messages as messages
class FleetClient(object):
    """Client for the Fleet API with related helper methods.

  If not provided, the default client is for the alpha (v1) track. This client
  is a thin wrapper around the base client, and does not handle any exceptions.

  Fields:
    release_track: The release track of the command [ALPHA, BETA, GA].
    client: The raw Fleet API client for the specified release track.
    messages: The matching messages module for the client.
    resourceless_waiter: A waiter.CloudOperationPollerNoResources for polling
      LROs that do not return a resource (like Deletes).
    fleet_waiter: A waiter.CloudOperationPoller for polling fleet LROs.
  """

    def __init__(self, release_track=base.ReleaseTrack.ALPHA):
        self.release_track = release_track
        self.client = util.GetClientInstance(release_track)
        self.messages = util.GetMessagesModule(release_track)
        self.resourceless_waiter = waiter.CloudOperationPollerNoResources(operation_service=self.client.projects_locations_operations)
        if release_track == base.ReleaseTrack.ALPHA:
            self.fleet_waiter = waiter.CloudOperationPoller(result_service=self.client.projects_locations_fleets, operation_service=self.client.projects_locations_operations)

    def GetFleet(self, project):
        """Gets a fleet resource from the Fleet API.

    Args:
      project: the project containing the fleet.

    Returns:
      A fleet resource

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        req = self.messages.GkehubProjectsLocationsFleetsGetRequest(name=util.FleetResourceName(project))
        return self.client.projects_locations_fleets.Get(req)

    def CreateFleet(self, req: messages.GkehubProjectsLocationsFleetsCreateRequest) -> messages.Operation:
        """Creates a fleet resource from the Fleet API.

    Args:
      req: An HTTP create request to be sent to the API server.

    Returns:
      A long-running operation to be polled till completion, or returned
      directly if user specifies async flag.

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        return self.client.projects_locations_fleets.Create(req)

    def DeleteFleet(self, req: messages.GkehubProjectsLocationsFleetsDeleteRequest) -> messages.Operation:
        """Deletes a fleet resource from the Fleet API.

    Args:
      req: An HTTP delete request to send to the API server.

    Returns:
      A long-running operation to be polled till completion, or returned
      directly if user specifies async flag.

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        return self.client.projects_locations_fleets.Delete(req)

    def UpdateFleet(self, req: messages.GkehubProjectsLocationsFleetsPatchRequest) -> messages.Operation:
        """Updates a fleet resource from the Fleet API.

    Args:
      req: An HTTP patch request to send to the API server.

    Returns:
      A long-running operation to be polled till completion, or returned
      directly if user specifies async flag.

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        return self.client.projects_locations_fleets.Patch(req)

    def ListFleets(self, project, organization):
        """Lists fleets in an organization.

    Args:
      project: the project to search.
      organization: the organization to search.

    Returns:
      A ListFleetResponse (list of fleets and next page token)

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        if organization:
            parent = util.FleetOrgParentName(organization)
        else:
            parent = util.FleetParentName(project)
        req = self.messages.GkehubProjectsLocationsFleetsListRequest(pageToken='', parent=parent)
        return list_pager.YieldFromList(self.client.projects_locations_fleets, req, field='fleets', batch_size_attribute=None)

    def GetScope(self, scope_path):
        """Gets a scope resource from the GKEHub API.

    Args:
      scope_path: Full resource path of the scope.

    Returns:
      A scope resource.

    Raises:
      apitools.base.py.HttpError: If the request returns an HTTP error
    """
        req = self.messages.GkehubProjectsLocationsScopesGetRequest(name=scope_path)
        return self.client.projects_locations_scopes.Get(req)

    def ListScopes(self, project):
        """Lists scopes in a project.

    Args:
      project: Project containing the scope.

    Returns:
      A ListScopesResponse (list of scopes and next page token).

    Raises:
      apitools.base.py.HttpError: If the request returns an HTTP error
    """
        parent = util.ScopeParentName(project)
        req = self.messages.GkehubProjectsLocationsScopesListRequest(pageToken='', parent=parent)
        return list_pager.YieldFromList(self.client.projects_locations_scopes, req, field='scopes', batch_size_attribute=None)

    def ListPermittedScopes(self, project):
        """Lists scopes in a project permitted to be viewed by the caller.

    Args:
      project: Project containing the scope.

    Returns:
      A ListPermittedScopesResponse (list of permitted scopes and next page
      token).

    Raises:
      apitools.base.py.HttpError: If the request returns an HTTP error
    """
        parent = util.ScopeParentName(project)
        req = self.messages.GkehubProjectsLocationsScopesListPermittedRequest(pageToken='', parent=parent)
        return list_pager.YieldFromList(self.client.projects_locations_scopes, req, method='ListPermitted', field='scopes', batch_size_attribute=None)

    def ListBoundMemberships(self, scope_path):
        """Lists memberships bound to a scope.

    Args:
      scope_path: Full resource path of the scope for listing bound memberships.

    Returns:
      A ListMembershipsResponse (list of memberships and next page token)

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        req = self.messages.GkehubProjectsLocationsScopesListMembershipsRequest(pageToken='', scopeName=scope_path)
        return list_pager.YieldFromList(self.client.projects_locations_scopes, req, method='ListMemberships', field='memberships', batch_size_attribute=None)

    def UpdateScope(self, scope_path, labels=None, namespace_labels=None, mask=''):
        """Updates a scope resource in the fleet.

    Args:
      scope_path: Full resource path of the scope.
      labels:  Labels for the resource.
      namespace_labels:  Namespace-level labels for the cluster namespace.
      mask: A mask of the fields to update.

    Returns:
      A longrunning operation for updating the namespace.
    """
        scope = self.messages.Scope(name=scope_path, labels=labels, namespaceLabels=namespace_labels)
        req = self.messages.GkehubProjectsLocationsScopesPatchRequest(scope=scope, name=scope_path, updateMask=mask)
        op = self.client.projects_locations_scopes.Patch(req)
        op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
        return waiter.WaitFor(waiter.CloudOperationPoller(self.client.projects_locations_scopes, self.client.projects_locations_operations), op_resource, 'Waiting for scope to be updated')

    def GetScopeNamespace(self, namespace_path):
        """Gets a namespace resource from the GKEHub API.

    Args:
      namespace_path: Full resource path of the namespace.

    Returns:
      A namespace resource.

    Raises:
      apitools.base.py.HttpError: If the request returns an HTTP error
    """
        req = self.messages.GkehubProjectsLocationsScopesNamespacesGetRequest(name=namespace_path)
        return self.client.projects_locations_scopes_namespaces.Get(req)

    def CreateScopeNamespace(self, name, namespace_path, parent, labels=None, namespace_labels=None):
        """Creates a namespace resource from the GKEHub API.

    Args:
      name: The namespace name.
      namespace_path: Full resource path of the namespace.
      parent: Full resource path of the scope containing the namespace.
      labels: labels for namespace resource.
      namespace_labels: Namespace-level labels for the cluster namespace.

    Returns:
      A namespace resource.

    Raises:
      apitools.base.py.HttpError: If the request returns an HTTP error.
    """
        namespace = self.messages.Namespace(name=namespace_path, scope='', labels=labels, namespaceLabels=namespace_labels)
        req = self.messages.GkehubProjectsLocationsScopesNamespacesCreateRequest(namespace=namespace, scopeNamespaceId=name, parent=parent)
        op = self.client.projects_locations_scopes_namespaces.Create(req)
        op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
        return waiter.WaitFor(waiter.CloudOperationPoller(self.client.projects_locations_scopes_namespaces, self.client.projects_locations_operations), op_resource, 'Waiting for namespace to be created')

    def DeleteScopeNamespace(self, namespace_path):
        """Deletes a namespace resource from the fleet.

    Args:
      namespace_path: Full resource path of the namespace.

    Returns:
      A long running operation for deleting the namespace.

    Raises:
      apitools.base.py.HttpError: If the request returns an HTTP error.
    """
        req = self.messages.GkehubProjectsLocationsScopesNamespacesDeleteRequest(name=namespace_path)
        op = self.client.projects_locations_scopes_namespaces.Delete(req)
        op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
        return waiter.WaitFor(waiter.CloudOperationPollerNoResources(self.client.projects_locations_operations), op_resource, 'Waiting for namespace to be deleted')

    def UpdateScopeNamespace(self, namespace_path, labels=None, namespace_labels=None, mask=''):
        """Updates a namespace resource in the fleet.

    Args:
      namespace_path: Full resource path of the namespace.
      labels:  Labels for the resource.
      namespace_labels:  Namespace-level labels for the cluster namespace.
      mask: A mask of the fields to update.

    Returns:
      A longrunning operation for updating the namespace.

    Raises:
    """
        namespace = self.messages.Namespace(name=namespace_path, scope='', labels=labels, namespaceLabels=namespace_labels)
        req = self.messages.GkehubProjectsLocationsScopesNamespacesPatchRequest(namespace=namespace, name=namespace_path, updateMask=mask)
        op = self.client.projects_locations_scopes_namespaces.Patch(req)
        op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
        return waiter.WaitFor(waiter.CloudOperationPoller(self.client.projects_locations_scopes_namespaces, self.client.projects_locations_operations), op_resource, 'Waiting for namespace to be updated')

    def ListScopeNamespaces(self, parent):
        """Lists namespaces in a project.

    Args:
      parent: Full resource path of the scope containing the namespace.

    Returns:
      A ListNamespaceResponse (list of namespaces and next page token).

    Raises:
      apitools.base.py.HttpError: If the request returns an HTTP error.
    """
        req = self.messages.GkehubProjectsLocationsScopesNamespacesListRequest(pageToken='', parent=parent)
        return list_pager.YieldFromList(self.client.projects_locations_scopes_namespaces, req, field='scopeNamespaces', batch_size_attribute=None)

    def GetNamespace(self, project, name):
        """Gets a namespace resource from the GKEHub API.

    Args:
      project: the project containing the namespace.
      name: the namespace name.

    Returns:
      A namespace resource

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        req = self.messages.GkehubProjectsLocationsNamespacesGetRequest(name=util.NamespaceResourceName(project, name))
        return self.client.projects_locations_namespaces.Get(req)

    def CreateNamespace(self, name, scope, project):
        """Creates a namespace resource from the GKEHub API.

    Args:
      name: the namespace name.
      scope: the scope containing the namespace.
      project: the project containing the namespace.

    Returns:
      A namespace resource

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        namespace = self.messages.Namespace(name=util.NamespaceResourceName(project, name), scope=scope)
        req = self.messages.GkehubProjectsLocationsNamespacesCreateRequest(namespace=namespace, namespaceId=name, parent=util.NamespaceParentName(project))
        op = self.client.projects_locations_namespaces.Create(req)
        op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
        return waiter.WaitFor(waiter.CloudOperationPoller(self.client.projects_locations_namespaces, self.client.projects_locations_operations), op_resource, 'Waiting for namespace to be created')

    def DeleteNamespace(self, project, name):
        """Deletes a namespace resource from the fleet.

    Args:
      project: the project containing the namespace.
      name: the name of the namespace.

    Returns:
      An operation

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        req = self.messages.GkehubProjectsLocationsNamespacesDeleteRequest(name=util.NamespaceResourceName(project, name))
        op = self.client.projects_locations_namespaces.Delete(req)
        op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
        return waiter.WaitFor(waiter.CloudOperationPollerNoResources(self.client.projects_locations_operations), op_resource, 'Waiting for namespace to be deleted')

    def UpdateNamespace(self, name, scope, project, mask):
        """Updates a namespace resource in the fleet.

    Args:
      name: the namespace name.
      scope: the scope containing the namespace.
      project: the project containing the namespace.
      mask: a mask of the fields to update.

    Returns:
      An operation

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        namespace = self.messages.Namespace(name=util.NamespaceResourceName(project, name), scope=scope)
        req = self.messages.GkehubProjectsLocationsNamespacesPatchRequest(namespace=namespace, name=util.NamespaceResourceName(project, name), updateMask=mask)
        op = self.client.projects_locations_namespaces.Patch(req)
        op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
        return waiter.WaitFor(waiter.CloudOperationPoller(self.client.projects_locations_namespaces, self.client.projects_locations_operations), op_resource, 'Waiting for namespace to be updated')

    def ListNamespaces(self, project):
        """Lists namespaces in a project.

    Args:
      project: the project to list namespaces from.

    Returns:
      A ListNamespaceResponse (list of namespaces and next page token)

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        req = self.messages.GkehubProjectsLocationsNamespacesListRequest(pageToken='', parent=util.NamespaceParentName(project))
        return list_pager.YieldFromList(self.client.projects_locations_namespaces, req, field='namespaces', batch_size_attribute=None)

    def GetRBACRoleBinding(self, name):
        """Gets an RBACRoleBinding resource from the GKEHub API.

    Args:
      name: the full rolebinding resource name.

    Returns:
      An RBACRoleBinding resource

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        req = self.messages.GkehubProjectsLocationsNamespacesRbacrolebindingsGetRequest(name=name)
        return self.client.projects_locations_namespaces_rbacrolebindings.Get(req)

    def CreateRBACRoleBinding(self, name, role, user, group):
        """Creates an RBACRoleBinding resource from the GKEHub API.

    Args:
      name: the full rbacrolebinding resource name.
      role: the role.
      user: the user.
      group: the group.

    Returns:
      An RBACRoleBinding resource

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
      calliope_exceptions.RequiredArgumentException: if a required field is
        missing
    """
        rolebinding = self.messages.RBACRoleBinding(name=name, role=self.messages.Role(predefinedRole=self.messages.Role.PredefinedRoleValueValuesEnum(role.upper())), user=user, group=group)
        resource = resources.REGISTRY.ParseRelativeName(name, 'gkehub.projects.locations.namespaces.rbacrolebindings', api_version='v1alpha')
        req = self.messages.GkehubProjectsLocationsNamespacesRbacrolebindingsCreateRequest(rBACRoleBinding=rolebinding, rbacrolebindingId=resource.Name(), parent=resource.Parent().RelativeName())
        return self.client.projects_locations_namespaces_rbacrolebindings.Create(req)

    def DeleteRBACRoleBinding(self, name):
        """Deletes an RBACRoleBinding resource from the fleet.

    Args:
      name: the resource name of the rolebinding.

    Returns:
      An operation

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        req = self.messages.GkehubProjectsLocationsNamespacesRbacrolebindingsDeleteRequest(name=name)
        return self.client.projects_locations_namespaces_rbacrolebindings.Delete(req)

    def UpdateRBACRoleBinding(self, name, user, group, role, mask):
        """Updates an RBACRoleBinding resource in the fleet.

    Args:
      name: the rolebinding name.
      user: the user.
      group: the group.
      role: the role.
      mask: a mask of the fields to update.

    Returns:
      An operation

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        rolebinding = self.messages.RBACRoleBinding(name=name, user=user, group=group, role=self.messages.Role(predefinedRole=self.messages.Role.PredefinedRoleValueValuesEnum(role.upper())))
        req = self.messages.GkehubProjectsLocationsNamespacesRbacrolebindingsPatchRequest(rBACRoleBinding=rolebinding, name=rolebinding.name, updateMask=mask)
        return self.client.projects_locations_namespaces_rbacrolebindings.Patch(req)

    def ListRBACRoleBindings(self, project, namespace):
        """Lists rolebindings in a namespace.

    Args:
      project: the project containing the namespace to list rolebindings from.
      namespace: the namespace to list rolebindings from.

    Returns:
      A ListNamespaceResponse (list of rolebindings and next page token)

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        req = self.messages.GkehubProjectsLocationsNamespacesRbacrolebindingsListRequest(pageToken='', parent=util.RBACRoleBindingParentName(project, namespace))
        return list_pager.YieldFromList(self.client.projects_locations_namespaces_rbacrolebindings, req, field='rbacrolebindings', batch_size_attribute=None)

    def GetScopeRBACRoleBinding(self, name):
        """Gets an ScopeRBACRoleBinding resource from the GKEHub API.

    Args:
      name: the full rolebinding resource name.

    Returns:
      An ScopeRBACRoleBinding resource

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        req = self.messages.GkehubProjectsLocationsScopesRbacrolebindingsGetRequest(name=name)
        return self.client.projects_locations_scopes_rbacrolebindings.Get(req)

    def CreateScopeRBACRoleBinding(self, name, role, user, group, labels=None):
        """Creates an ScopeRBACRoleBinding resource from the GKEHub API.

    Args:
      name: the full Scoperbacrolebinding resource name.
      role: the role.
      user: the user.
      group: the group.
      labels: labels for the RBACRoleBinding resource.

    Returns:
      An ScopeRBACRoleBinding resource

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
      calliope_exceptions.RequiredArgumentException: if a required field is
        missing
    """
        rolebinding = self.messages.RBACRoleBinding(name=name, role=self.messages.Role(predefinedRole=self.messages.Role.PredefinedRoleValueValuesEnum(role.upper())), user=user, group=group, labels=labels)
        resource = resources.REGISTRY.ParseRelativeName(name, 'gkehub.projects.locations.scopes.rbacrolebindings', api_version='v1alpha')
        req = self.messages.GkehubProjectsLocationsScopesRbacrolebindingsCreateRequest(rBACRoleBinding=rolebinding, rbacrolebindingId=resource.Name(), parent=resource.Parent().RelativeName())
        op = self.client.projects_locations_scopes_rbacrolebindings.Create(req)
        op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
        return waiter.WaitFor(waiter.CloudOperationPoller(self.client.projects_locations_scopes_rbacrolebindings, self.client.projects_locations_operations), op_resource, 'Waiting for rbacrolebinding to be created')

    def DeleteScopeRBACRoleBinding(self, name):
        """Deletes an ScopeRBACRoleBinding resource from the fleet.

    Args:
      name: the resource name of the rolebinding.

    Returns:
      An operation

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        req = self.messages.GkehubProjectsLocationsScopesRbacrolebindingsDeleteRequest(name=name)
        op = self.client.projects_locations_scopes_rbacrolebindings.Delete(req)
        op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
        return waiter.WaitFor(waiter.CloudOperationPollerNoResources(self.client.projects_locations_operations), op_resource, 'Waiting for rbacrolebinding to be deleted')

    def UpdateScopeRBACRoleBinding(self, name, user, group, role, labels, mask):
        """Updates an ScopeRBACRoleBinding resource in the fleet.

    Args:
      name: the rolebinding name.
      user: the user.
      group: the group.
      role: the role.
      labels: labels for the RBACRoleBinding resource.
      mask: a mask of the fields to update.

    Returns:
      An operation

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        rolebinding = self.messages.RBACRoleBinding(name=name, user=user, group=group, role=self.messages.Role(predefinedRole=self.messages.Role.PredefinedRoleValueValuesEnum(role.upper())), labels=labels)
        req = self.messages.GkehubProjectsLocationsScopesRbacrolebindingsPatchRequest(rBACRoleBinding=rolebinding, name=rolebinding.name, updateMask=mask)
        op = self.client.projects_locations_scopes_rbacrolebindings.Patch(req)
        op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
        return waiter.WaitFor(waiter.CloudOperationPoller(self.client.projects_locations_scopes_rbacrolebindings, self.client.projects_locations_operations), op_resource, 'Waiting for rbacrolebinding to be updated')

    def ListScopeRBACRoleBindings(self, project, scope):
        """Lists rolebindings in a scope.

    Args:
      project: the project containing the scope to list rolebindings from.
      scope: the scope to list rolebindings from.

    Returns:
      A ListscopeResponse (list of rolebindings and next page token)

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        req = self.messages.GkehubProjectsLocationsScopesRbacrolebindingsListRequest(pageToken='', parent=util.ScopeRBACRoleBindingParentName(project, scope))
        return list_pager.YieldFromList(self.client.projects_locations_scopes_rbacrolebindings, req, field='rbacrolebindings', batch_size_attribute=None)

    def GetMembershipBinding(self, name):
        """Gets a Membership-Binding resource from the GKEHub API.

    Args:
      name: the full membership-binding resource name.

    Returns:
      A Membership-Binding resource

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        req = self.messages.GkehubProjectsLocationsMembershipsBindingsGetRequest(name=name)
        return self.client.projects_locations_memberships_bindings.Get(req)

    def CreateMembershipBinding(self, name, scope, labels=None):
        """Creates a Membership-Binding resource from the GKEHub API.

    Args:
      name: the full binding resource name.
      scope: the Scope to be associated with Binding.
      labels: labels for the membership binding resource

    Returns:
      A Membership-Binding resource

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
      calliope_exceptions.RequiredArgumentException: if a required field is
        missing
    """
        binding = self.messages.MembershipBinding(name=name, scope=scope, labels=labels)
        resource = resources.REGISTRY.ParseRelativeName(name, 'gkehub.projects.locations.memberships.bindings', api_version='v1alpha')
        req = self.messages.GkehubProjectsLocationsMembershipsBindingsCreateRequest(membershipBinding=binding, membershipBindingId=resource.Name(), parent=resource.Parent().RelativeName())
        op = self.client.projects_locations_memberships_bindings.Create(req)
        op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
        return waiter.WaitFor(waiter.CloudOperationPoller(self.client.projects_locations_memberships_bindings, self.client.projects_locations_operations), op_resource, 'Waiting for membership binding to be created')

    def ListMembershipBindings(self, project, membership, location='global'):
        """Lists Bindings in a Membership.

    Args:
      project: the project containing the Membership to list Bindings from.
      membership: the Membership to list Bindings from.
      location: the Membrship location to list Bindings

    Returns:
      A ListMembershipBindingResponse (list of bindings and next page token)

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        req = self.messages.GkehubProjectsLocationsMembershipsBindingsListRequest(pageToken='', parent=util.MembershipBindingParentName(project, membership, location, self.release_track))
        return list_pager.YieldFromList(self.client.projects_locations_memberships_bindings, req, field='membershipBindings', batch_size_attribute=None)

    def UpdateMembershipBinding(self, name, scope, labels, mask):
        """Updates a Membership-Binding resource.

    Args:
      name: the Binding name.
      scope: the Scope associated with binding.
      labels: the labels for the Membership Binding resource.
      mask: a mask of the fields to update.

    Returns:
      An operation

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        binding = self.messages.MembershipBinding(name=name, scope=scope, labels=labels)
        req = self.messages.GkehubProjectsLocationsMembershipsBindingsPatchRequest(membershipBinding=binding, name=binding.name, updateMask=mask)
        op = self.client.projects_locations_memberships_bindings.Patch(req)
        op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
        return waiter.WaitFor(waiter.CloudOperationPoller(self.client.projects_locations_memberships_bindings, self.client.projects_locations_operations), op_resource, 'Waiting for membership binding to be updated')

    def DeleteMembershipBinding(self, name):
        """Deletes a Membership-Binding resource.

    Args:
      name: the resource name of the Binding.

    Returns:
      An operation

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        req = self.messages.GkehubProjectsLocationsMembershipsBindingsDeleteRequest(name=name)
        op = self.client.projects_locations_memberships_bindings.Delete(req)
        op_resource = resources.REGISTRY.ParseRelativeName(op.name, collection='gkehub.projects.locations.operations')
        return waiter.WaitFor(waiter.CloudOperationPollerNoResources(self.client.projects_locations_operations), op_resource, 'Waiting for membership binding to be deleted')

    def GetMembershipRbacRoleBinding(self, name):
        """Gets a Membership RBAC RoleBinding resource from the GKEHub API.

    Args:
      name: the full Membership RBAC RoleBinding resource name.

    Returns:
      A Membership RBAC Role Binding resource.

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error.
    """
        req = self.messages.GkehubProjectsLocationsMembershipsRbacrolebindingsGetRequest(name=name)
        return self.client.projects_locations_memberships_rbacrolebindings.Get(req)

    def CreateMembershipRbacRoleBinding(self, name, role, user, group):
        """Creates a Membership RBAC RoleBinding resource from the GKEHub API.

    Args:
      name: the full Membership RBAC Role Binding resource name.
      role: the role for the RBAC policies.
      user: the user to apply the RBAC policies for.
      group: the group to apply the RBAC policies for.

    Returns:
      A Membership RBAC Role Binding resource.

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error.
    """
        rolebinding = self.messages.RBACRoleBinding(name=name, role=self.messages.Role(predefinedRole=self.messages.Role.PredefinedRoleValueValuesEnum(role.upper())), user=user, group=group)
        resource = resources.REGISTRY.ParseRelativeName(name, 'gkehub.projects.locations.memberships.rbacrolebindings', api_version='v1alpha')
        req = self.messages.GkehubProjectsLocationsMembershipsRbacrolebindingsCreateRequest(rBACRoleBinding=rolebinding, rbacrolebindingId=resource.Name(), parent=resource.Parent().RelativeName())
        return self.client.projects_locations_memberships_rbacrolebindings.Create(req)

    def DeleteMembershipRbacRoleBinding(self, name):
        """Deletes a Membership RBAC RoleBinding resource.

    Args:
      name: the resource name of the Membership RBAC RoleBinding.

    Returns:
      A long running operation for the deletion.

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error.
    """
        req = self.messages.GkehubProjectsLocationsMembershipsRbacrolebindingsDeleteRequest(name=name)
        return self.client.projects_locations_memberships_rbacrolebindings.Delete(req)

    def GenerateMembershipRbacRoleBindingYaml(self, name, role, user, group):
        """Gets YAML containing RBAC policies for a membership RBAC role binding.

    Args:
      name: the full Membership RBAC Role Binding resource name.
      role: the role for the RBAC policies.
      user: the user to apply the RBAC policies for.
      group: the group to apply the RBAC policies for.

    Returns:
      YAML text containing RBAC policies for a membership RBAC rolebinding.

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error.
    """
        rolebinding = self.messages.RBACRoleBinding(name=name, role=self.messages.Role(predefinedRole=self.messages.Role.PredefinedRoleValueValuesEnum(role.upper())), user=user, group=group)
        resource = resources.REGISTRY.ParseRelativeName(name, 'gkehub.projects.locations.memberships.rbacrolebindings', api_version='v1alpha')
        req = self.messages.GkehubProjectsLocationsMembershipsRbacrolebindingsGenerateMembershipRBACRoleBindingYAMLRequest(rBACRoleBinding=rolebinding, rbacrolebindingId=resource.Name(), parent=resource.Parent().RelativeName())
        return self.client.projects_locations_memberships_rbacrolebindings.GenerateMembershipRBACRoleBindingYAML(req)

    def CreateRollout(self, req: messages.GkehubProjectsLocationsRolloutsCreateRequest) -> messages.Operation:
        """Creates a rollout resource from the Fleet rollout API.

    Args:
      req: An HTTP create rollout request to be sent to the API server.

    Returns:
      A long-running operation.

    Raises:
      apitools.base.py.HttpError: if the request returns an HTTP error
    """
        return self.client.projects_locations_rollouts.Create(req)

    def DescribeRollout(self, req: messages.GkehubProjectsLocationsRolloutsGetRequest) -> messages.Rollout:
        """Describes a fleet rollout."""
        return self.client.projects_locations_rollouts.Get(req)

    def ListRollouts(self, req: messages.GkehubProjectsLocationsRolloutsListRequest, page_size=None, limit=None) -> Generator[messages.Rollout, None, None]:
        """Lists fleet rollouts."""
        return list_pager.YieldFromList(self.client.projects_locations_rollouts, req, field='rollouts', batch_size=page_size, limit=limit, batch_size_attribute='pageSize')

    def PauseRollout(self, req: messages.GkehubProjectsLocationsRolloutsPauseRequest) -> messages.Operation:
        """Pause a fleet rollout."""
        return self.client.projects_locations_rollouts.Pause(req)

    def ResumeRollout(self, req: messages.GkehubProjectsLocationsRolloutsResumeRequest) -> messages.Operation:
        """Resume a fleet rollout."""
        return self.client.projects_locations_rollouts.Resume(req)

    def DeleteRollout(self, req: messages.GkehubProjectsLocationsRolloutsDeleteRequest) -> messages.Operation:
        """Delete a fleet rollout."""
        return self.client.projects_locations_rollouts.Delete(req)