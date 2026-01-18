import platform as _platform
import sys as _sys
from tensorflow.python import autograph
from tensorflow.python.ops import array_grad
from tensorflow.python.ops import cudnn_rnn_grad
from tensorflow.python.ops import data_flow_grad
from tensorflow.python.ops import manip_grad
from tensorflow.python.ops import math_grad
from tensorflow.python.ops import random_grad
from tensorflow.python.ops import rnn_grad
from tensorflow.python.ops import sparse_grad
from tensorflow.python.ops import state_grad
from tensorflow.python.ops import tensor_array_grad
from tensorflow.python.ops.array_ops import *  # pylint: disable=redefined-builtin
from tensorflow.python.ops.check_ops import *
from tensorflow.python.ops.clip_ops import *
from tensorflow.python.ops.special_math_ops import *
from tensorflow.python.ops.cond import cond
from tensorflow.python.ops.confusion_matrix import confusion_matrix
from tensorflow.python.ops.control_flow_assert import Assert
from tensorflow.python.ops.control_flow_case import case
from tensorflow.python.ops.control_flow_ops import group
from tensorflow.python.ops.control_flow_ops import no_op
from tensorflow.python.ops.control_flow_ops import tuple  # pylint: disable=redefined-builtin
from tensorflow.python.eager import wrap_function
from tensorflow.python.ops.while_loop import while_loop
from tensorflow.python.ops.batch_ops import *
from tensorflow.python.ops.critical_section_ops import *
from tensorflow.python.ops.data_flow_ops import *
from tensorflow.python.ops.functional_ops import *
from tensorflow.python.ops.gradients import *
from tensorflow.python.ops.histogram_ops import *
from tensorflow.python.ops.init_ops import *
from tensorflow.python.ops.io_ops import *
from tensorflow.python.ops.linalg_ops import *
from tensorflow.python.ops.linalg import adjoint_registrations as _adjoint_registrations
from tensorflow.python.ops.linalg import cholesky_registrations as _cholesky_registrations
from tensorflow.python.ops.linalg import inverse_registrations as _inverse_registrations
from tensorflow.python.ops.linalg import linear_operator_algebra as _linear_operator_algebra
from tensorflow.python.ops.linalg import matmul_registrations as _matmul_registrations
from tensorflow.python.ops.linalg import solve_registrations as _solve_registrations
from tensorflow.python.ops.logging_ops import Print
from tensorflow.python.ops.logging_ops import get_summary_op
from tensorflow.python.ops.logging_ops import timestamp
from tensorflow.python.ops.lookup_ops import initialize_all_tables
from tensorflow.python.ops.lookup_ops import tables_initializer
from tensorflow.python.ops.manip_ops import *
from tensorflow.python.ops.math_ops import *  # pylint: disable=redefined-builtin
from tensorflow.python.ops.numerics import *
from tensorflow.python.ops.parsing_ops import *
from tensorflow.python.ops.partitioned_variables import *
from tensorflow.python.ops.proto_ops import *
from tensorflow.python.ops.ragged import ragged_batch_gather_ops
from tensorflow.python.ops.ragged import ragged_batch_gather_with_default_op
from tensorflow.python.ops.ragged import ragged_bincount_ops
from tensorflow.python.ops.ragged import ragged_check_ops
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow.python.ops.ragged import ragged_dispatch as _ragged_dispatch
from tensorflow.python.ops.ragged import ragged_embedding_ops
from tensorflow.python.ops.ragged import ragged_image_ops
from tensorflow.python.ops.ragged import ragged_operators as _ragged_operators
from tensorflow.python.ops.ragged import ragged_squeeze_op
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.ops.random_ops import *
from tensorflow.python.ops.script_ops import py_func
from tensorflow.python.ops.session_ops import *
from tensorflow.python.ops.sort_ops import *
from tensorflow.python.ops.sparse_ops import *
from tensorflow.python.ops.state_ops import assign
from tensorflow.python.ops.state_ops import assign_add
from tensorflow.python.ops.state_ops import assign_sub
from tensorflow.python.ops.state_ops import count_up_to
from tensorflow.python.ops.state_ops import scatter_add
from tensorflow.python.ops.state_ops import scatter_div
from tensorflow.python.ops.state_ops import scatter_mul
from tensorflow.python.ops.state_ops import scatter_sub
from tensorflow.python.ops.state_ops import scatter_min
from tensorflow.python.ops.state_ops import scatter_max
from tensorflow.python.ops.state_ops import scatter_update
from tensorflow.python.ops.state_ops import scatter_nd_add
from tensorflow.python.ops.state_ops import scatter_nd_sub
from tensorflow.python.ops.state_ops import scatter_nd_update
from tensorflow.python.ops.stateless_random_ops import *
from tensorflow.python.ops.string_ops import *
from tensorflow.python.ops.template import *
from tensorflow.python.ops.tensor_array_ops import *
from tensorflow.python.ops.variable_scope import *  # pylint: disable=redefined-builtin
from tensorflow.python.ops.variables import *
from tensorflow.python.ops.parallel_for.control_flow_ops import vectorized_map
from tensorflow.python.compiler.tensorrt import trt_convert as trt
Import names of Tensor Flow standard Ops.