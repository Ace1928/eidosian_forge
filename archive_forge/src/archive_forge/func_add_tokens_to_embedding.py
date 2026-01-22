import argparse
import transformers
import torch
def add_tokens_to_embedding(added_special_tokens, embedding):
    new_token_embeddings = torch.mean(embedding.to(torch.float32), dim=0, keepdim=True).to(embedding.dtype)
    new_token_embeddings = new_token_embeddings.expand(len(added_special_tokens), -1)
    return torch.cat([embedding, new_token_embeddings], dim=0)