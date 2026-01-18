import atexit
import ssl
from pyVim.connect import Disconnect, SmartStubAdapter, VimSessionOrientedStub
from pyVmomi import vim
from ray.autoscaler._private.vsphere.utils import Constants, singleton_client

        This function will return the vSphere object.
        The argument for `vimtype` can be "vim.VM", "vim.Host", "vim.Datastore", etc.
        Then either the name or the object id need to be provided.
        To check all such object information, you can go to the managed object board
        page of your vCenter Server, such as: https://<your_vc_ip/mob
        