from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.privateca import flags
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.command_lib.util.args import labels_util
def CreateCAFromArgs(args, is_subordinate):
    """Creates a GA CA object from CA create flags.

  Args:
    args: The parser that contains the flag values.
    is_subordinate: If True, a subordinate CA is returned, otherwise a root CA.

  Returns:
    A tuple for the CA to create with (CA object, CA ref, issuer).
  """
    client = privateca_base.GetClientInstance(api_version='v1')
    messages = privateca_base.GetMessagesModule(api_version='v1')
    ca_ref, source_ca_ref, issuer_ref = _ParseCAResourceArgs(args)
    pool_ref = ca_ref.Parent()
    source_ca = None
    if source_ca_ref:
        source_ca = client.projects_locations_caPools_certificateAuthorities.Get(messages.PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesGetRequest(name=source_ca_ref.RelativeName()))
        if not source_ca:
            raise exceptions.InvalidArgumentException('--from-ca', 'The provided source CA could not be retrieved.')
    ca_pool = client.projects_locations_caPools.Get(messages.PrivatecaProjectsLocationsCaPoolsGetRequest(name=pool_ref.RelativeName()))
    keyspec = flags.ParseKeySpec(args)
    if ca_pool.tier == messages.CaPool.TierValueValuesEnum.DEVOPS and keyspec.cloudKmsKeyVersion:
        raise exceptions.InvalidArgumentException('--kms-key-version', 'The DevOps tier does not support user-specified KMS keys.')
    subject_config = messages.SubjectConfig(subject=messages.Subject(), subjectAltName=messages.SubjectAltNames())
    if args.IsSpecified('subject'):
        subject_config.subject = flags.ParseSubject(args)
    elif source_ca:
        subject_config.subject = source_ca.config.subjectConfig.subject
    if flags.SanFlagsAreSpecified(args):
        subject_config.subjectAltName = flags.ParseSanFlags(args)
    elif source_ca:
        subject_config.subjectAltName = source_ca.config.subjectConfig.subjectAltName
    flags.ValidateSubjectConfig(subject_config, is_ca=True)
    x509_parameters = flags.ParseX509Parameters(args, is_ca_command=True)
    if source_ca and (not flags.X509ConfigFlagsAreSpecified(args)):
        x509_parameters = source_ca.config.x509Config
    lifetime = flags.ParseValidityFlag(args)
    if source_ca and (not args.IsSpecified('validity')):
        lifetime = source_ca.lifetime
    labels = labels_util.ParseCreateArgs(args, messages.CertificateAuthority.LabelsValue)
    ski = flags.ParseSubjectKeyId(args, messages)
    new_ca = messages.CertificateAuthority(type=messages.CertificateAuthority.TypeValueValuesEnum.SUBORDINATE if is_subordinate else messages.CertificateAuthority.TypeValueValuesEnum.SELF_SIGNED, lifetime=lifetime, config=messages.CertificateConfig(subjectConfig=subject_config, x509Config=x509_parameters, subjectKeyId=ski), keySpec=keyspec, gcsBucket=None, labels=labels)
    return (new_ca, ca_ref, issuer_ref)