from typing import Callable, Tuple, Any
def _register_pytree_with_jax(pytree_type: type, flatten_fn: FlattenFn, unflatten_fn: UnflattenFn):

    def jax_unflatten(aux, parameters):
        return unflatten_fn(parameters, aux)
    jax_tree_util.register_pytree_node(pytree_type, flatten_fn, jax_unflatten)