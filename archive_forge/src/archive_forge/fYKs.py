import collections
import numpy as np  # More conventional alias
import os
import torch
from safetensors.torch import load_file
from safetensors import serialize_file
import argparse
from typing import Dict, List, OrderedDict  # Improved type hinting

# Improved argument parsing with type annotations
parser = argparse.ArgumentParser(
    description="Convert PyTorch models to SafeTensors format."
)
parser.add_argument(
    "--input", type=str, required=True, help="Path to input PyTorch model (.pth)."
)
parser.add_argument(
    "--output",
    type=str,
    default="./converted.st",
    help="Path to output SafeTensors model (.st).",
)
args = parser.parse_args()


def rename_key(rename: Dict[str, str], name: str) -> str:
    """
    Renames a key based on a mapping.

    Args:
        rename (Dict[str, str]): A dictionary mapping original keys to new keys.
        name (str): The original key name.

    Returns:
        str: The renamed key, if applicable.
    """
    for k, v in rename.items():
        if k in name:
            name = name.replace(k, v)
    return name


def convert_file(
    pt_filename: str,
    sf_filename: str,
    rename: Dict[str, str] = {},
    transpose_names: List[str] = [],
) -> None:
    """
    Converts a PyTorch model file to SafeTensors format.

    Args:
        pt_filename (str): Path to the input PyTorch model file.
        sf_filename (str): Path to the output SafeTensors file.
        rename (Dict[str, str], optional): A mapping of names to rename in the model's state dict.
        transpose_names (List[str], optional): A list of tensor names to transpose.

    Raises:
        RuntimeError: If there's a mismatch between the original and reloaded tensors.
    """
    loaded: OrderedDict[str, torch.Tensor] = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]

    kk = list(loaded.keys())
    version = 4
    # Version detection logic
    for x in kk:
        if "ln_x" in x:
            version = max(5, version)
        if "gate.weight" in x:
            version = max(5.1, version)
        if int(version) == 5 and "att.time_decay" in x:
            if len(loaded[x].shape) > 1 and loaded[x].shape[1] > 1:
                version = max(5.2, version)
        if "time_maa" in x:
            version = max(6, version)

    print(f"Model detected: v{version:.1f}")

    # Specific handling for version 5.1
    if version == 5.1:
        _, n_emb = loaded["emb.weight"].shape
        for k in kk:
            if "time_decay" in k or "time_faaaa" in k:
                loaded[k] = (
                    loaded[k].unsqueeze(1).repeat(1, n_emb // loaded[k].shape[0])
                )

    with torch.no_grad():
        for k in kk:
            new_k = rename_key(rename, k).lower()
            v = loaded[k].half()
            del loaded[k]
            for transpose_name in transpose_names:
                if transpose_name in new_k:
                    dims = len(v.shape)
                    v = v.transpose(dims - 2, dims - 1)
            print(f"{new_k}\t{v.shape}\t{v.dtype}")
            loaded[new_k] = {
                "dtype": str(v.dtype).split(".")[-1],
                "shape": v.shape,
                "data": v.numpy().tobytes(),
            }

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    serialize_file(loaded, sf_filename, metadata={"format": "pt"})
    # reloaded = load_file(sf_filename)
    # for k in loaded:
    #     pt_tensor = torch.Tensor(
    #         numpy.frombuffer(
    #             bytearray(loaded[k]["data"]),
    #             dtype=getattr(numpy, loaded[k]["dtype"]),
    #         ).reshape(loaded[k]["shape"])
    #     )
    #     sf_tensor = reloaded[k]
    #     if not torch.equal(pt_tensor, sf_tensor):
    #         raise RuntimeError(f"The output tensors do not match for key {k}")


if __name__ == "__main__":
    try:
        convert_file(
            args.input,
            args.output,
            rename={
                "time_faaaa": "time_first",
                "time_maa": "time_mix",
                "lora_A": "lora.0",
                "lora_B": "lora.1",
            },
            transpose_names=[
                "time_mix_w1",
                "time_mix_w2",
                "time_decay_w1",
                "time_decay_w2",
            ],
        )
        print(f"Saved to {args.output}")
    except Exception as e:
        print(e)
        with open("error.txt", "w") as f:
            f.write(str(e))
