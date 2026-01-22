import os
import torch
from safetensors.torch import save_file
import argparse
from typing import Dict, List, OrderedDict

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
    """
    return rename.get(name, name)


def process_tensor(tensor: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    """
    Processes a tensor, converting it to half precision and optionally transposing it.
    """
    tensor = tensor.half()
    if transpose:
        tensor = tensor.transpose(-2, -1)
    return tensor


def convert_file(
    pt_filename: str,
    sf_filename: str,
    rename: Dict[str, str] = {},
    transpose_names: List[str] = [],
):
    """
    Converts a PyTorch model file to SafeTensors format.
    """
    loaded: OrderedDict[str, torch.Tensor] = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]

    processed_tensors = {}
    for name, tensor in loaded.items():
        new_name = rename_key(rename, name).lower()
        transpose = any(
            transpose_name in new_name for transpose_name in transpose_names
        )
        processed_tensors[new_name] = process_tensor(tensor, transpose)

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(processed_tensors, sf_filename, metadata={"format": "pt"})
    print(f"Saved to {sf_filename}")


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
    except Exception as e:
        print(e)
        with open("error.txt", "w") as f:
            f.write(str(e))
