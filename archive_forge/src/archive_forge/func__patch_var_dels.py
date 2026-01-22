from functools import cached_property
from numba.core import ir, analysis, transforms, ir_utils
def _patch_var_dels(self, internal_dead_map, escaping_dead_map, extend_lifetimes=False):
    """
        Insert delete in each block
        """
    for offset, ir_block in self.func_ir.blocks.items():
        internal_dead_set = internal_dead_map[offset].copy()
        delete_pts = []
        for stmt in reversed(ir_block.body[:-1]):
            live_set = set((v.name for v in stmt.list_vars()))
            dead_set = live_set & internal_dead_set
            for T, def_func in ir_extension_insert_dels.items():
                if isinstance(stmt, T):
                    done_dels = def_func(stmt, dead_set)
                    dead_set -= done_dels
                    internal_dead_set -= done_dels
            delete_pts.append((stmt, dead_set))
            internal_dead_set -= dead_set
        body = []
        lastloc = ir_block.loc
        del_store = []
        for stmt, delete_set in reversed(delete_pts):
            if extend_lifetimes:
                lastloc = ir_block.body[-1].loc
            else:
                lastloc = stmt.loc
            if not isinstance(stmt, ir.Del):
                body.append(stmt)
            for var_name in sorted(delete_set, reverse=True):
                delnode = ir.Del(var_name, loc=lastloc)
                if extend_lifetimes:
                    del_store.append(delnode)
                else:
                    body.append(delnode)
        if extend_lifetimes:
            body.extend(del_store)
        body.append(ir_block.body[-1])
        ir_block.body = body
        escape_dead_set = escaping_dead_map[offset]
        for var_name in sorted(escape_dead_set):
            ir_block.prepend(ir.Del(var_name, loc=ir_block.body[0].loc))