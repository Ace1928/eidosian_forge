import logging
import torch
from xformers import _is_triton_available
from xformers.ops import masked_matmul
def _initialize_triton_ops(self):
    block_size = self.__values.shape[-1]
    self.__sparse_dot_sdd = blocksparse_matmul(self.__layout, block_size, 'sdd', trans_a=False, trans_b=True, device=self.__layout.device)
    self.__sparse_dot_dsd = blocksparse_matmul(self.__layout, block_size, 'dsd', trans_a=False, trans_b=False, device=self.__layout.device)
    self.__sparse_softmax = blocksparse_softmax(self.__layout, block_size, device=self.__layout.device)