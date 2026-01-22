import argparse
import os
from pyproj import __proj_version__, __version__, _show_versions
from pyproj.aoi import BBox
from pyproj.datadir import get_data_dir, get_user_data_dir
from pyproj.sync import (
def _parse_sync_command(args):
    """
    Handle sync command arguments
    """
    if not any((args.bbox, args.list_files, args.all, args.source_id, args.area_of_use, args.file)):
        sync_parser.print_help()
        return
    if args.all and any((args.bbox, args.list_files, args.source_id, args.area_of_use, args.file)):
        raise RuntimeError("Cannot use '--all' with '--list-files', '--source-id','--area-of-use', '--bbox', or '--file'.")
    bbox = None
    if args.bbox is not None:
        west, south, east, north = args.bbox.split(',')
        bbox = BBox(west=float(west), south=float(south), east=float(east), north=float(north))
    if args.target_directory and args.system_directory:
        raise RuntimeError('Cannot set both --target-directory and --system-directory.')
    target_directory = args.target_directory
    if args.system_directory:
        target_directory = get_data_dir().split(os.path.sep)[0]
    elif not target_directory:
        target_directory = get_user_data_dir(True)
    grids = get_transform_grid_list(source_id=args.source_id, area_of_use=args.area_of_use, filename=args.file, bbox=bbox, spatial_test=args.spatial_test, include_world_coverage=not args.exclude_world_coverage, include_already_downloaded=args.include_already_downloaded, target_directory=target_directory)
    if args.list_files:
        print('filename | source_id | area_of_use')
        print('----------------------------------')
    else:
        endpoint = get_proj_endpoint()
    for grid in grids:
        if args.list_files:
            print(grid['properties']['name'], grid['properties']['source_id'], grid['properties'].get('area_of_use'), sep=' | ')
        else:
            filename = grid['properties']['name']
            _download_resource_file(file_url=f'{endpoint}/{filename}', short_name=filename, directory=target_directory, verbose=args.verbose, sha256=grid['properties']['sha256sum'])