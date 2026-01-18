from collections import defaultdict
import click
def validate_multilayer_file_index(files, layerdict):
    """
    Ensure file indexes provided in the --layer option are valid
    """
    for key in layerdict.keys():
        if key not in [str(k) for k in range(1, len(files) + 1)]:
            layer = key + ':' + layerdict[key][0]
            raise click.BadParameter(f'Layer {layer} does not exist')