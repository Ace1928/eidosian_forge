def _restore_vars_from_nlp_block_saved_values(nlp_util_block):
    for var, old_value in nlp_util_block.initial_var_values.items():
        if not var.fixed and var.is_continuous():
            if old_value is not None:
                if var.has_lb() and old_value < var.lb:
                    old_value = var.lb
                if var.has_ub() and old_value > var.ub:
                    old_value = var.ub
                var.set_value(old_value)