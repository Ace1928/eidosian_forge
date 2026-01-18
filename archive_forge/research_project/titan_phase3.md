
### Phase 3: Integrating Attention Mechanisms and Persistent Memory
8. **Persistent Memory Integration**  
   - Add persistent memory parameters as learnable embeddings:
     - Define buffers or parameters for `[p_1, ..., p_{N_p}]` inside the module or as a separate layer.
   - Modify input preprocessing:
     - Concatenate persistent tokens to the beginning of input sequences before processing.

9. **Attention Mechanism Wrappers**  
   - Create new classes or functions for Titan variants:
     - **MAC Variant**:
       - Implement retrieval of historical context using the memory (`Equation 21`).
       - Concatenate persistent tokens, retrieved memory, and current input (`Equation 22`).
       - Apply full attention on this enriched sequence.
       - Update memory using attention outputs (`Equations 24–25`).
     - **MAG Variant**:
       - Prepend persistent tokens and apply sliding window attention (`Equations 26–28`).
       - Combine attention and memory outputs using a gating mechanism.
     - **MAL Variant**:
       - Prepend persistent tokens, pass through the memory layer, then apply sliding window attention (`Equations 29–31`).
   - Each variant can initially reuse the existing `TitanMemory` logic and later integrate more features.

---
