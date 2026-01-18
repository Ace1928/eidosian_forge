
### Phase 2: Core Feature Enhancements Toward Paper Concepts
5. **Adaptive Forgetting Mechanism**  
   - Extend `_update_memory` to:
     - Compute an adaptive `α_t` based on current input or state via learned parameters
     - Use this dynamic `α_t` in the forgetting equation
 
6. **Deep Memory Architecture**  
   - Design a separate deep memory module (e.g., `DeepTitanMemory`) that:
     - Extends and compliments the linear memory update with a multi-layer perceptron network.
     - Incorporates additional layers to increase representational capacity.
   - Integrate this new module within the Titan framework as an alternative and complimentary module to the current memory mechanism.

7. **Memory Retrieval Function**  
   - Add a method, `retrieve_memory(query: torch.Tensor) -> torch.Tensor`, which:
     - Loads, if needed, offloaded tensors using `_load_from_disk()`
     - Projects the query with `self.query_projection`
     - Retrieves memory output without modifying state, using current `memory_matrix`

---
