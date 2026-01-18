from typing import Union
def var_pp_to_str(self, var_name, pp_var_name):
    """WIP."""
    var_name_str = self.var_name_to_str(var_name)
    pp_var_name_str = self.var_name_to_str(pp_var_name)
    if var_name_str == pp_var_name_str:
        return f'{var_name_str}'
    return f'{var_name_str} / {pp_var_name_str}'