from typing import Union
class MapLabeller(BaseLabeller):
    """WIP."""

    def __init__(self, var_name_map=None, dim_map=None, coord_map=None, model_name_map=None):
        """WIP."""
        self.var_name_map = {} if var_name_map is None else var_name_map
        self.dim_map = {} if dim_map is None else dim_map
        self.coord_map = {} if coord_map is None else coord_map
        self.model_name_map = {} if model_name_map is None else model_name_map

    def dim_coord_to_str(self, dim, coord_val, coord_idx):
        """WIP."""
        dim_str = self.dim_map.get(dim, dim)
        coord_str = self.coord_map.get(dim, {}).get(coord_val, coord_val)
        return super().dim_coord_to_str(dim_str, coord_str, coord_idx)

    def var_name_to_str(self, var_name):
        """WIP."""
        var_name_str = self.var_name_map.get(var_name, var_name)
        return super().var_name_to_str(var_name_str)

    def model_name_to_str(self, model_name):
        """WIP."""
        model_name_str = self.var_name_map.get(model_name, model_name)
        return super().model_name_to_str(model_name_str)