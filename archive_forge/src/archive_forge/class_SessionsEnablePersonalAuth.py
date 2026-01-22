from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import files
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class SessionsEnablePersonalAuth(base.Command):
    """Enable personal auth on a session."""
    detailed_help = {'EXAMPLES': '          To enable personal auth session on a `my-session` session in the `us-central1` location, run:\n\n            $ {command} my-session --location=us-central1\n          '}

    @staticmethod
    def Args(parser):
        """Method called by Calliope to register flags for this command.

    Args:
      parser: An argparser parser used to register flags.
    """
        dataproc = dp.Dataproc()
        flags.AddSessionResourceArg(parser, 'enable-personal-auth-session', dataproc.api_version)
        flags.AddPersonalAuthSessionArgs(parser)

    def _get_session_key(self, dataproc, get_session_request, session_name, key_type):
        """Get Session public key.

    Args:
      dataproc: The API client for calling into the Dataproc API.
      get_session_request: A DataprocProjectsLocationsSessionsGetRequest object
        to get session.
      session_name: The name of the session.
      key_type: Public keys can be of type RSA or ECIES.

    Returns:
      The public key for the session.
    """
        session = dataproc.client.projects_locations_sessions.Get(get_session_request)
        try:
            session_public_keys = session.runtimeInfo.publicKeys.keys
        except Exception:
            raise exceptions.PersonalAuthError('Unable to retrieve public keys for session {}.'.format(session_name))
        for session_public_key in session_public_keys:
            if session_public_key.type == dataproc.messages.Key.TypeValueValuesEnum.ECIES:
                session_ecies_key = session_public_key.content
            elif session_public_key.type == dataproc.messages.Key.TypeValueValuesEnum.RSA:
                session_rsa_key = session_public_key.content
        if key_type == 'ECIES' and session_ecies_key:
            session_key = session_ecies_key
        else:
            session_key = session_rsa_key
        if not session_key:
            raise exceptions.PersonalAuthError('The session {} does not support personal auth.'.format(session_name))
        return session_key

    def inject_credentials(self, dataproc, session_name, session_id, session_key, access_boundary_json, operation_poller, openssl_executable):
        """Get credentials and inject them into session.

    Args:
      dataproc: The API client for calling into the Dataproc API.
      session_name: The name of the session.
      session_id: Relative name of the session. Format:
        'projects/{}/locations/{}/session/{}'
      session_key: The public key for the session.
      access_boundary_json: The JSON-formatted access boundary.
      operation_poller: Poller for the cloud operation.
      openssl_executable: The path to the openssl executable.
    """
        downscoped_token = util.GetCredentials(access_boundary_json)
        if not downscoped_token:
            raise exceptions.PersonalAuthError('Failure getting credentials to inject into {}'.format(session_name))
        credentials_ciphertext = util.PersonalAuthUtils().EncryptWithPublicKey(session_key, downscoped_token, openssl_executable)
        inject_operation = _inject_encrypted_credentials(dataproc, session_id, credentials_ciphertext)
        if inject_operation:
            waiter.WaitFor(operation_poller, inject_operation)

    def Run(self, args):
        message = 'A personal authentication session will propagate your personal credentials to the dataproc session, so make sure you trust the session and the user who created it.'
        console_io.PromptContinue(message=message, cancel_on_no=True, cancel_string='Enabling personal auth session aborted by user')
        dataproc = dp.Dataproc(self.ReleaseTrack())
        session_ref = args.CONCEPTS.session.Parse()
        session_id = session_ref.RelativeName()
        session_name = session_ref.Name()
        project = properties.VALUES.core.project.Get()
        request = dataproc.messages.DataprocProjectsLocationsSessionsGetRequest(name=session_id)
        if args.access_boundary:
            with files.FileReader(args.access_boundary) as abf:
                access_boundary_json = abf.read()
        else:
            access_boundary_json = flags.ProjectGcsObjectsAccessBoundary(project)
        session_key_type = 'ECIES' if util.PersonalAuthUtils().IsTinkLibraryInstalled() else 'RSA'
        session_key = self._get_session_key(dataproc, request, session_name, session_key_type)
        openssl_executable = None
        if session_key_type == 'RSA':
            openssl_executable = args.openssl_command
            if not openssl_executable:
                try:
                    openssl_executable = files.FindExecutableOnPath('openssl')
                except ValueError:
                    log.fatal('Could not find openssl on your system. The enable-session command requires openssl to be installed.')
        operation_poller = waiter.CloudOperationPollerNoResources(dataproc.client.projects_regions_operations, lambda operation: operation.name)
        try:
            with progress_tracker.ProgressTracker('Injecting initial credentials into the session {}'.format(session_name), autotick=True):
                self.inject_credentials(dataproc, session_name, session_id, session_key, access_boundary_json, operation_poller, openssl_executable)
            if not args.refresh_credentials:
                return
            with progress_tracker.ProgressTracker('Periodically refreshing credentials for session {}. This will continue running until the command is interrupted'.format(session_name), autotick=True):
                try:
                    session_key = self._get_session_key(dataproc, request, session_name, session_key_type)
                    failure_count = 0
                    while failure_count < 3:
                        try:
                            time.sleep(30)
                            self.inject_credentials(dataproc, session_name, session_id, session_key, access_boundary_json, operation_poller, openssl_executable)
                            failure_count = 0
                        except ValueError as err:
                            log.error(err)
                            failure_count += 1
                    raise exceptions.PersonalAuthError('Credential injection failed three times in a row, giving up...')
                except (console_io.OperationCancelledError, KeyboardInterrupt):
                    log.status.write('Refreshing credentials injection interrupted by user.')
                    return
        except exceptions.PersonalAuthError as err:
            log.error(err)
            return