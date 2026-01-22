from ..base import BaseInterface, BaseInterfaceInputSpec, traits
from ...utils.imagemanip import copy_header as _copy_header
class CopyHeaderInputSpec(BaseInterfaceInputSpec):
    copy_header = traits.Bool(desc='Copy headers of the input image into the output image')