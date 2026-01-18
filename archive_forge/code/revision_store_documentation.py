from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils
from ... import revision as _mod_revision
from ...bzr import inventory
from ...bzr.inventorytree import InventoryTreeChange
Load a revision by applying a delta to a (CHK)Inventory.

        :param rev: the Revision
        :param basis_inv: the basis Inventory or CHKInventory
        :param inv_delta: the inventory delta
        :param signature: signing information
        :param text_provider: a callable expecting a file_id parameter
            that returns the text for that file-id
        :param parents_provider: a callable expecting a file_id parameter
            that return the list of parent-ids for that file-id
        :param inventories_provider: a callable expecting a repository and
            a list of revision-ids, that returns:
              * the list of revision-ids present in the repository
              * the list of inventories for the revision-id's,
                including an empty inventory for the missing revisions
            If None, a default implementation is provided.
        