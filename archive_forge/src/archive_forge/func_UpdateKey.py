from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.container.binauthz import apis
from googlecloudsdk.api_lib.container.binauthz import util
from googlecloudsdk.command_lib.container.binauthz import exceptions
from googlecloudsdk.command_lib.kms import maps as kms_maps
def UpdateKey(self, attestor_ref, pubkey_id, pgp_pubkey_content=None, comment=None):
    """Update a key on an attestor.

    Args:
      attestor_ref: ResourceSpec, The attestor to be updated.
      pubkey_id: The ID of the key to update.
      pgp_pubkey_content: The contents of the public key file.
      comment: The comment on the public key.

    Returns:
      The updated public key.

    Raises:
      NotFoundError: If an expected public key could not be located by ID.
      InvalidStateError: If multiple public keys matched the provided ID.
      InvalidArgumentError: If a non-PGP key is updated with pgp_pubkey_content.
    """
    attestor = self.Get(attestor_ref)
    existing_keys = [public_key for public_key in self.GetNoteAttr(attestor).publicKeys if public_key.id == pubkey_id]
    if not existing_keys:
        raise exceptions.NotFoundError('No matching public key found on attestor [{}]'.format(attestor.name))
    if len(existing_keys) > 1:
        raise exceptions.InvalidStateError('Multiple matching public keys found on attestor [{}]'.format(attestor.name))
    existing_key = existing_keys[0]
    if pgp_pubkey_content is not None:
        if not existing_key.asciiArmoredPgpPublicKey:
            raise exceptions.InvalidArgumentError('Cannot update a non-PGP PublicKey with a PGP public key')
        existing_key.asciiArmoredPgpPublicKey = pgp_pubkey_content
    if comment is not None:
        existing_key.comment = comment
    updated_attestor = self.client.projects_attestors.Update(attestor)
    return next((public_key for public_key in self.GetNoteAttr(updated_attestor).publicKeys if public_key.id == pubkey_id))